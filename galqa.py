#!/usr/bin/env python
# coding: utf-8

# # Preliminary Research - Law Granularity Aware QA
# ## Retriever-Reranker using TF-IDF and Siamese BERT

# ========================================================================================================
# Prerequisites (Libraries, Logging, and Functions)
# ========================================================================================================
# ====================================================
# Import Libraries and Start Logging
# ====================================================
import numpy as np
import time
import torch
import random
import logging
import json
from collections import Counter
import sys
import nltk
import argparse
import copy
from LawComponent import LawComponent
import re
import copy
from elasticsearch import Elasticsearch
import pandas as pd
import uuid
import json
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sentence_transformers import losses
from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.model_selection import train_test_split

nltk.download('punkt')

logging.basicConfig(filename='./galqa-siamese.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', force=True)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info("START Experiment")

# ====================================================
# Hyperparameters
# ====================================================

class HPARAMS:

        # query hyper params
        text_match_boost = 1.0
        text_phrase_match_boost = 1.0
        chapter_match_boost = 1.0
        chapter_phrase_match_boost = 1.0
        use_question_to_statement = False
        use_keyword_law_filter = False

        # retriever param
        top_k = 3

        # reranker model
        reranker_mode = 'BiEncoder' # 'BiEncoder', 'CrossEncoder'
        reranker_model = 'indolem/indobert-base-uncased'

        # reranker training hyper params
        epochs=1 
        batch_size=1
        learning_rate = 2e-5  
        test_size=0.25
        k_random_negative_examples = 3 
        prepend_component_type = False 
        resample_test_data=False 
        rerank_relevance_only=False 
        log_train_results = True 

        # BiEncoder-exclusive hyper param
        max_siamese_train_size = -1
        
        # indexing hyper params
        chapter_title_only_for_chapter=False 
        include_map=False 
        use_stop_removal_stemming=False 
        article_only_index=False 
        use_stop_only=False 
        use_stem_only=False 
        

argparser = argparse.ArgumentParser(description='Process hyper-parameters')       
for attribute, default_value in vars(HPARAMS).items():
    if not attribute.startswith('__'):
        argparser.add_argument(f'--{attribute}', type=type(default_value), default=default_value)
args = argparser.parse_args()

for attribute in vars(HPARAMS).keys():
    if not attribute.startswith('__'):
        setattr(HPARAMS, attribute, getattr(args, attribute))

logging.info("Hyperparameters: " + json.dumps({k: v for k, v in HPARAMS.__dict__.items() if not k.startswith('__')},
                                              sort_keys=True, indent=4))

# ====================================================
# Configurations
# ====================================================

class CFG:

  BASE_DIR = "/workspace/thesis/"
  # BASE_DIR = "/workspace/"
  ELASTICSEARCH_HOST = 'http://localhost:9200'
  RANDOM_SEED = 123

  # Mapping types for dataset
  col_types = {
    'nomorPasal': str,
    'nomorAyat': 'Int64',
    'nomorHuruf': str
  }

  # Default missing values for dataset
  fill_values_dataset = {
    'nomorPeraturan': 0,
    'nomorBab': 0,
    'nomorPasal': 0,
    'nomorAyat': 0,
    'nomorHuruf': 0
  }

  # Mapping types for QA dataset
  test_data_types = {
    "Chapter": "Int64",
    "Article": str,
    "Subsection": "Int64",
    "Letter (1st level)": str
  }

  # Translate granularity for QA dataset
  answer_granularity_dict = {
    'Bab': 'Chapter',
    'Pasal': 'Article',
    'Ayat': 'Subsection',
    'Huruf': 'Letter'
  }

  # Default missing values for QA dataset
  fill_values_qa_dataset = {
    'Chapter': 0,
    'Article': 0,
    'Subsection': 0,
    'Letter (1st level)': 0
  }

  base_query = {
    "query": {
      "bool": {
        "should": [
          {
              "match": {
                  "text": {
                      "query": None,
                      "boost": HPARAMS.text_match_boost
                  }
              }
          },
          {
              "match": {
                  "chapterTitle": {
                      "query": None,
                      "boost": HPARAMS.chapter_match_boost
                  }
              }
          },
          {
              "match_phrase": {
                  "text": {
                      "query": None,
                      "boost": HPARAMS.text_phrase_match_boost
                  }
              }
          },
          {
              "match_phrase": {
                  "chapterTitle": {
                      "query": None,
                      "boost": HPARAMS.chapter_phrase_match_boost
                  }
              }
          }
        ],
        "minimum_should_match": 1
      }
    }
  }

  test_indexes = [162, 118, 51, 113, 24, 171, 97, 159, 181, 117, 119, 164, 29, 25, 136, 72, 176, 123, 19, 156, 132, 199, 6, 9, 128, 195, 86, 3, 107, 154, 170, 17, 53, 95, 36, 68, 76, 58, 92, 81, 50, 83, 203, 46, 143, 111, 47, 62, 184, 191, 101]

# ====================================================
# Seed everything
# ====================================================

def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)

seed_everything(seed_value = CFG.RANDOM_SEED)

# ====================================================
# Indexing Utility Function
# ====================================================

# Keyword-based filter
def extract_definition_keyword(definition_text: str):
  try:
    find = re.search('(?P<base_term>([A-Z][a-z\s\/]*)+) adalah', definition_text)
    find_2 = re.sub(' yang selanjutnya (disingkat|disebut) ', '/', find.group('base_term'))
    results = find_2.split('/')
    return results
  except AttributeError:
    return None

def get_keywords_from_query(query: str):
  return [i for i in reverse_keywords_dict if i.lower() in query.lower()]

# ====================================================
# Querying Utility Function
# ====================================================

def question_to_statement(question: str):
  if not HPARAMS.use_question_to_statement:
    return question
  pattern = "(?P<qWord>kapan|(ber|meng|si)?apa|(di|ke) ?(mana)|bagaimana) ?(saja)? ?(kah)?"

  result = re.sub(pattern, "", question.lower())
  result = result.replace("?", "")
  result = re.sub("\s+", " ", result)
  result = result.strip()
  return result

# given chapter, get an article, subsection, and letter (if possible)
def chapter_find_in_line(qa_data):
  in_line_indexes = []
  law_component = LawComponent.from_answer_granularity_row(qa_data)
  
  # get single article
  article_results = df.query('nomorBab == @nomor_bab and nomorPeraturan == @nomor_peraturan and type == "pasal"', engine='python')
  if (len(article_results) > 0):
    # negative_text_passages.append(article_results.iloc[0]['isiTeks'])
    in_line_indexes += article_results.index.to_list()

  # get single subsection
  subsection_results = df.query('nomorBab == @nomor_bab and nomorPeraturan == @nomor_peraturan and type == "ayat"', engine='python')
  if (len(subsection_results) > 0):
    # negative_text_passages.append(subsection_results.iloc[0]['isiTeks'])
    in_line_indexes += subsection_results.index.to_list()

  # get single letter
  letter_results = df.query('nomorBab == @nomor_bab and nomorPeraturan == @nomor_peraturan and (type == "huruf-ayat" or type == "huruf-pasal")', engine='python')
  if (len(letter_results) > 0):
    # negative_text_passages.append(letter_results.iloc[0]['isiTeks'])
    in_line_indexes += letter_results.index.to_list()
  
  # return negative_text_passages
  return in_line_indexes

# given article, get the chapter, subsection and letter
def article_find_in_line(qa_data):
  in_line_indexes = []
  law_component = LawComponent.from_answer_granularity_row(qa_data)
  nomor_bab = law_component.chapter
  nomor_pasal = str(law_component.article)
  nomor_peraturan = law_component.law_number

  # get single chapter
  chapter_results = df.query('nomorBab == @nomor_bab and nomorPeraturan == @nomor_peraturan and type == "bab"', engine='python')
  if (len(chapter_results) > 0):
    # negative_text_passages.append(chapter_results.iloc[0]['isiTeks'])
    in_line_indexes += chapter_results.index.to_list()

  # get single subsection
  subsection_results = df.query('nomorPeraturan == @nomor_peraturan and nomorPasal == @nomor_pasal and type == "ayat"', engine='python')
  if (len(subsection_results) > 0):
    # negative_text_passages.append(subsection_results.iloc[0]['isiTeks'])
    in_line_indexes += subsection_results.index.to_list()  

  # get single letter
  letter_results = df.query('nomorPeraturan == @nomor_peraturan and nomorPasal == @nomor_pasal and (type == "huruf-ayat" or type == "huruf-pasal")', engine='python')
  if (len(letter_results) > 0):
    # negative_text_passages.append(letter_results.iloc[0]['isiTeks'])
    in_line_indexes += letter_results.index.to_list()
  
  # return negative_text_passages
  return in_line_indexes

# given subsection, get chapter, article, and letter (if possible)
def subsection_find_in_line(qa_data):
  in_line_indexes = []
  law_component = LawComponent.from_answer_granularity_row(qa_data)
  nomor_bab = law_component.chapter
  nomor_pasal = str(law_component.article)
  nomor_ayat = law_component.subsection
  nomor_peraturan = law_component.law_number

  # get single chapter
  chapter_results = df.query('nomorBab == @nomor_bab and nomorPeraturan == @nomor_peraturan and type == "bab"', engine='python')
  if (len(chapter_results) > 0):
    # negative_text_passages.append(chapter_results.iloc[0]['isiTeks'])
    in_line_indexes += chapter_results.index.to_list()

  # get single article
  article_results = df.query('nomorPeraturan == @nomor_peraturan and nomorPasal == @nomor_pasal and type == "pasal"', engine='python')
  if (len(article_results) > 0):
    # negative_text_passages.append(article_results.iloc[0]['isiTeks'])
    in_line_indexes += article_results.index.to_list()

  # get single letter
  letter_results = df.query('nomorPeraturan == @nomor_peraturan and nomorPasal == @nomor_pasal and nomorAyat == @nomor_ayat and (type == "huruf-ayat" or type == "huruf-pasal")', engine='python')
  if (len(letter_results) > 0):
    # negative_text_passages.append(letter_results.iloc[0]['isiTeks'])
    in_line_indexes += letter_results.index.to_list()
  
  # return negative_text_passages
  return in_line_indexes

# given letter, get chapter, article and subsection
def letter_find_in_line(qa_data):
  in_line_indexes = []
  law_component = LawComponent.from_answer_granularity_row(qa_data)
  nomor_bab = law_component.chapter
  nomor_pasal = str(law_component.article)
  nomor_ayat = law_component.subsection
  nomor_huruf = str(law_component.letter)
  nomor_peraturan = law_component.law_number

  # get single chapter
  chapter_results = df.query('nomorBab == @nomor_bab and nomorPeraturan == @nomor_peraturan and type == "bab"', engine='python')
  if (len(chapter_results) > 0):
    # negative_text_passages.append(chapter_results.iloc[0]['isiTeks'])
    in_line_indexes += chapter_results.index.to_list()

  # get single article
  article_results = df.query('nomorPeraturan == @nomor_peraturan and nomorPasal == @nomor_pasal and type == "pasal"', engine='python')
  if (len(article_results) > 0):
    # negative_text_passages.append(article_results.iloc[0]['isiTeks'])
    in_line_indexes += article_results.index.to_list()

  # get single subsection
  subsection_results = df.query('nomorPeraturan == @nomor_peraturan and nomorPasal == @nomor_pasal and nomorAyat == @nomor_ayat and type == "ayat"', engine='python')
  if (len(subsection_results) > 0):
    # negative_text_passages.append(subsection_results.iloc[0]['isiTeks'])
    in_line_indexes += subsection_results.index.to_list()
  
  # return negative_text_passages
  return in_line_indexes

def find_in_line_indexes(qa_data):
  if (qa_data['Answer Granularity'] == 'Chapter'):
    return chapter_find_in_line(qa_data)
  elif (qa_data['Answer Granularity'] == 'Article'):
    return article_find_in_line(qa_data)
  elif (qa_data['Answer Granularity'] == 'Subsection'):
    return subsection_find_in_line(qa_data)
  elif (qa_data['Answer Granularity'] == 'Letter'):
    return letter_find_in_line(qa_data)
  return []

def get_test_result(row):
  question = row['Question']
  query = copy.deepcopy(CFG.base_query)
  query['query']['bool']['should'][0]['match']['text']['query'] = question_to_statement(question)
  query['query']['bool']['should'][1]['match']['chapterTitle']['query'] = question_to_statement(question)
  query['query']['bool']['should'][2]['match_phrase']['text']['query'] = question_to_statement(question)
  query['query']['bool']['should'][3]['match_phrase']['chapterTitle']['query'] = question_to_statement(question)
  
  # keywords & law filter
  keywords = get_keywords_from_query(question)
  if len(keywords) > 0 and HPARAMS.use_keyword_law_filter:
    law_uris = set().union(*[reverse_keywords_dict[k] for k in keywords])
    query['query']['bool']['filter'] = {
        "terms": {
          "lawUri": list(law_uris)
        }
      }
  

  # search
  ap_result = 0
  if (HPARAMS.include_map):
    result = es.search(
      index="index_redundant" + "_" + str(cur_timestamp), 
    #       body=query
      query=query['query'],
      size=3000
    )
    
    expected_lc = LawComponent.from_answer_granularity_row(row)
    in_line_lcs = [LawComponent.from_uri(uri) for uri in df.loc[row['in_line_indexes']]['uri'].to_list()]
    
    rank = None
    score = 0
    
    df_es_results_data = []
    
    for i, hit in enumerate(result['hits']['hits']):
        actual_lc = LawComponent.from_uri(hit['_source']['uri'])
        if actual_lc == expected_lc:
            rank = i+1
            score = hit['_score']
            break
            
    df_es_results_data.append({
        'question': question,
        'rank': rank,
        'score': score,
        'type': 'expected'
    })
    
    # in-line
    for in_line_lc in in_line_lcs:
        rank = None
        score = 0
    
        for i, hit in enumerate(result['hits']['hits']):
            actual_lc = LawComponent.from_uri(hit['_source']['uri'])
            if actual_lc == in_line_lc:
                rank = i+1
                score = hit['_score']
                break
            
        df_es_results_data.append({
            'question': question,
            'rank': rank,
            'score': score,
            'type': 'in-line'
        })
    
    # calculate mAP
    df_es_results = pd.DataFrame(df_es_results_data)
    df_es_results = df_es_results.sort_values('rank')
    rank_count = len(df_es_results)
    df_es_results = df_es_results.dropna(subset=['rank'])
    ranks = sorted(df_es_results['rank'].to_list())
    
    prec_sum = 0
    for i,rank_i in enumerate(ranks):
        if rank_i == None:
            continue
        prec_sum += ((i+1) / rank_i)
    ap_result = prec_sum/rank_count
    
  else:
    result = es.search(
      index="index_redundant" + "_" + str(cur_timestamp), 
    #       body=query
      query=query['query']
    )

  try:
    top_1_test_result = result['hits']['hits'][0]
    top_k_test_result = result['hits']['hits'][:HPARAMS.top_k]
  except IndexError as e:
    top_1_test_result = {}
    top_k_test_result = {}
  
  return top_1_test_result, top_k_test_result, ap_result

# ====================================================
# Evaluation Utility Function
# ====================================================

def get_tokens(passage):
  return nltk.word_tokenize(passage)

def extract_results(results):
  
  '''
  Used for wrapping function to either return passage text only or with 
  the type
  '''

  if (HPARAMS.prepend_component_type):
    return results.iloc[0]['type'] + ' [GRANULARITY] ' + results.iloc[0]['isiTeks']
  else:
    return results.iloc[0]['isiTeks']

def get_text(row):
  law_component = LawComponent.from_answer_granularity_row(row)
  try:
    results = df.query('law_component == @law_component')
    return extract_results(results)
  except IndexError:
    return None

def evaluate_retriever(row):
  lc_expected = LawComponent.from_answer_granularity_row(row.to_dict())
  all_correct = False
  article_correct = False
  answer_in_top_k = False
  f1_score = 0.0
  try:
    lc_actual = LawComponent.from_uri(row['top_1']['_source']['uri'])
    all_correct = lc_expected == lc_actual
    article_correct = lc_expected.is_article_equal(lc_actual)
    f1_score = compute_qa_f1(row['expected_text'], row['top_1']['_source']['text'])
    
    for ans in row['top_k']:
        lc_actual_i = LawComponent.from_uri(ans['_source']['uri'])
        if lc_actual_i == lc_expected:
            answer_in_top_k = True
            break
        
  except KeyError:
    pass

  return {"all_correct":all_correct, 
          "article_correct":article_correct,
          "answer_in_top_k":answer_in_top_k,
          "f1_score":f1_score}

def train_test_split_from_retriever(df):
  df_train = df.drop(CFG.test_indexes).copy(deep=True)
  df_test = df.loc[CFG.test_indexes].copy(deep=True)

  if (HPARAMS.resample_test_data):
    df_train, df_test = train_test_split(
        df, test_size=HPARAMS.test_size, stratify=df['Answer Granularity'],
        random_state=CFG.RANDOM_SEED)

    df_train = df_train.copy(deep=True)
    df_test = df_test.copy(deep=True)
  
  return df_train, df_test

# ====================================================
# Metrics
# ====================================================

def compute_qa_f1(actual, pred):

  if type(actual) != str or type(pred) != str:
    return 0

  pred_tokens = get_tokens(pred)
  truth_tokens = get_tokens(actual)
  
  # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
  if len(pred_tokens) == 0 or len(truth_tokens) == 0:
    return int(pred_tokens == truth_tokens)
  
  common_tokens = set(pred_tokens) & set(truth_tokens)
  
  # if there are no common tokens then f1 = 0
  if len(common_tokens) == 0:
    return 0
  
  prec = len(common_tokens) / len(pred_tokens)
  rec = len(common_tokens) / len(truth_tokens)
  
  return 2 * (prec * rec) / (prec + rec)

# ====================================================
# Reranker Utility Functions
# ====================================================

# given chapter, get an article, subsection, and letter (if possible)
def chapter_find_negative_examples(qa_data):
  negative_text_passages = []
  law_component = LawComponent.from_answer_granularity_row(qa_data)
  nomor_bab = law_component.chapter
  nomor_peraturan = law_component.law_number
  
  # get single article
  article_results = df.query('nomorBab == @nomor_bab and nomorPeraturan == @nomor_peraturan and type == "pasal"', engine='python')
  if (len(article_results) > 0):
    # negative_text_passages.append(article_results.iloc[0]['isiTeks'])
    negative_text_passages.append(extract_results(article_results))

  # get single subsection
  subsection_results = df.query('nomorBab == @nomor_bab and nomorPeraturan == @nomor_peraturan and type == "ayat"', engine='python')
  if (len(subsection_results) > 0):
    # negative_text_passages.append(subsection_results.iloc[0]['isiTeks'])
    negative_text_passages.append(extract_results(subsection_results))

  # get single letter
  letter_results = df.query('nomorBab == @nomor_bab and nomorPeraturan == @nomor_peraturan and (type == "huruf-ayat" or type == "huruf-pasal")', engine='python')
  if (len(letter_results) > 0):
    # negative_text_passages.append(letter_results.iloc[0]['isiTeks'])
    negative_text_passages.append(extract_results(letter_results))
  
  # return negative_text_passages
  return law_component, negative_text_passages

# given article, get the chapter, subsection and letter
def article_find_negative_examples(qa_data):
  negative_text_passages = []
  law_component = LawComponent.from_answer_granularity_row(qa_data)
  nomor_bab = law_component.chapter
  nomor_pasal = str(law_component.article)
  nomor_peraturan = law_component.law_number

  # get single chapter
  chapter_results = df.query('nomorBab == @nomor_bab and nomorPeraturan == @nomor_peraturan and type == "bab"', engine='python')
  if (len(chapter_results) > 0):
    # negative_text_passages.append(chapter_results.iloc[0]['isiTeks'])
    negative_text_passages.append(extract_results(chapter_results))

  # get single subsection
  subsection_results = df.query('nomorPeraturan == @nomor_peraturan and nomorPasal == @nomor_pasal and type == "ayat"', engine='python')
  if (len(subsection_results) > 0):
    # negative_text_passages.append(subsection_results.iloc[0]['isiTeks'])
    negative_text_passages.append(extract_results(subsection_results))    

  # get single letter
  letter_results = df.query('nomorPeraturan == @nomor_peraturan and nomorPasal == @nomor_pasal and (type == "huruf-ayat" or type == "huruf-pasal")', engine='python')
  if (len(letter_results) > 0):
    # negative_text_passages.append(letter_results.iloc[0]['isiTeks'])
    negative_text_passages.append(extract_results(letter_results))
  
  # return negative_text_passages
  return law_component, negative_text_passages

# given subsection, get chapter, article, and letter (if possible)
def subsection_find_negative_examples(qa_data):
  negative_text_passages = []
  law_component = LawComponent.from_answer_granularity_row(qa_data)
  nomor_bab = law_component.chapter
  nomor_pasal = str(law_component.article)
  nomor_ayat = law_component.subsection
  nomor_peraturan = law_component.law_number

  # get single chapter
  chapter_results = df.query('nomorBab == @nomor_bab and nomorPeraturan == @nomor_peraturan and type == "bab"', engine='python')
  if (len(chapter_results) > 0):
    # negative_text_passages.append(chapter_results.iloc[0]['isiTeks'])
    negative_text_passages.append(extract_results(chapter_results))

  # get single article
  article_results = df.query('nomorPeraturan == @nomor_peraturan and nomorPasal == @nomor_pasal and type == "pasal"', engine='python')
  if (len(article_results) > 0):
    # negative_text_passages.append(article_results.iloc[0]['isiTeks'])
    negative_text_passages.append(extract_results(article_results))

  # get single letter
  letter_results = df.query('nomorPeraturan == @nomor_peraturan and nomorPasal == @nomor_pasal and nomorAyat == @nomor_ayat and (type == "huruf-ayat" or type == "huruf-pasal")', engine='python')
  if (len(letter_results) > 0):
    # negative_text_passages.append(letter_results.iloc[0]['isiTeks'])
    negative_text_passages.append(extract_results(letter_results))
  
  # return negative_text_passages
  return law_component, negative_text_passages

# given letter, get chapter, article and subsection
def letter_find_negative_examples(qa_data):
  negative_text_passages = []
  law_component = LawComponent.from_answer_granularity_row(qa_data)
  nomor_bab = law_component.chapter
  nomor_pasal = str(law_component.article)
  nomor_ayat = law_component.subsection
  nomor_huruf = str(law_component.letter)
  nomor_peraturan = law_component.law_number

  # get single chapter
  chapter_results = df.query('nomorBab == @nomor_bab and nomorPeraturan == @nomor_peraturan and type == "bab"', engine='python')
  if (len(chapter_results) > 0):
    # negative_text_passages.append(chapter_results.iloc[0]['isiTeks'])
    negative_text_passages.append(extract_results(chapter_results))

  # get single article
  article_results = df.query('nomorPeraturan == @nomor_peraturan and nomorPasal == @nomor_pasal and type == "pasal"', engine='python')
  if (len(article_results) > 0):
    # negative_text_passages.append(article_results.iloc[0]['isiTeks'])
    negative_text_passages.append(extract_results(article_results))

  # get single subsection
  subsection_results = df.query('nomorPeraturan == @nomor_peraturan and nomorPasal == @nomor_pasal and nomorAyat == @nomor_ayat and type == "ayat"', engine='python')
  if (len(subsection_results) > 0):
    # negative_text_passages.append(subsection_results.iloc[0]['isiTeks'])
    negative_text_passages.append(extract_results(subsection_results))
  
  # return negative_text_passages
  return law_component, negative_text_passages

def find_negative_examples(qa_data):
  if (qa_data['Answer Granularity'] == 'Chapter'):
    return chapter_find_negative_examples(qa_data)
  elif (qa_data['Answer Granularity'] == 'Article'):
    return article_find_negative_examples(qa_data)
  elif (qa_data['Answer Granularity'] == 'Subsection'):
    return subsection_find_negative_examples(qa_data)
  elif (qa_data['Answer Granularity'] == 'Letter'):
    return letter_find_negative_examples(qa_data)
  return None, []

def sample_random_negative_examples(lc: LawComponent, k: int):
  if k==0:
    return []  # early `return` to avoid random state from being changed

  df_other_idxs = df.query('law_component != @lc').index.to_list()
  sampled_idxs = random.sample(df_other_idxs, k)
  negative_text_passages = []

  for _,row in df.loc[sampled_idxs].iterrows():
    negative_text_passages.append(row['isiTeks'])
  
  return negative_text_passages

def log_model_results(df_train, df_test):
  if (HPARAMS.log_train_results):
    logging.info("[TRAIN] Reranked QA F1 Score Average {:.2f}".format(df_train['f1_score'].mean() * 100))
    logging.info("[TRAIN] Reranked Exact Match {:.2f}".format(len(df_train.query('top_1_rerank_match == True')) / len(df_train) * 100))
    logging.info("[TRAIN] Reranked Article Match {:.2f}".format(len(df_train.query('top_1_rerank_article_match == True')) / len(df_train.query('`Answer Granularity` != "Chapter"')) * 100))
    logging.info("[TRAIN] Percent Answer in Top-k {:.2f}".format(len(df_train.query('answer_in_top_k == True')) / len(df_train) * 100))
    logging.info("[TRAIN] Isolated Reranking Exact Match {:.2f}".format(len(df_train.query('top_1_rerank_match == True')) / len(df_train.query('answer_in_top_k == True')) * 100))
    
  logging.info("[TEST] Reranked QA F1 Score Average {:.2f}".format(df_test['f1_score'].mean() * 100))
  logging.info("[TEST] Reranked Exact Match {:.2f}".format(len(df_test.query('top_1_rerank_match == True')) / len(df_test) * 100))
  logging.info("[TEST] Reranked Article Match {:.2f}".format(len(df_test.query('top_1_rerank_article_match == True')) / len(df_test.query('`Answer Granularity` != "Chapter"')) * 100))
  logging.info("[TEST] Percent Answer in Top-k {:.2f}".format(len(df_test.query('answer_in_top_k == True')) / len(df_test) * 100))
  logging.info("[TEST] Isolated Reranking Exact Match {:.2f}".format(len(df_test.query('top_1_rerank_match == True')) / len(df_test.query('answer_in_top_k == True')) * 100))

  answer_granularities = test_data_redundant['Answer Granularity'].unique()

  logging.info('Reranked Result Per Category')
  logging.info('============================')

  granularity_score_sum_train = 0
  granularity_score_sum_test = 0

  for i in answer_granularities:

    df_ag = df_train.query('`Answer Granularity` == @i')

    all_correct_acc = len(df_ag.query('top_1_rerank_match == True')) / len(df_ag)
    granularity_score_sum_train += all_correct_acc

    if (HPARAMS.log_train_results):
      logging.info('[TRAIN] Answer Granularity: {}'.format(i))
      logging.info('[TRAIN] QA F1 Score Average: {:.2f}'.format(df_ag['f1_score'].mean() * 100))
      logging.info('[TRAIN] Exact Match {:.2f}'.format(all_correct_acc*100))
      logging.info('[TRAIN] Test Count: ({}/{})'.format(len(df_ag.query('top_1_rerank_match == True')), len(df_ag)))
      logging.info('---------')

    df_ag = df_test.query('`Answer Granularity` == @i')
    
    all_correct_acc = len(df_ag.query('top_1_rerank_match == True')) / len(df_ag)
    granularity_score_sum_test += all_correct_acc

    logging.info('[TEST] Answer Granularity: {}'.format(i))
    logging.info('[TEST] QA F1 Score Average: {:.2f}'.format(df_ag['f1_score'].mean() * 100))
    logging.info('[TEST] Exact Match {:.2f}'.format(all_correct_acc*100))
    logging.info('[TEST] Test Count: ({}/{})'.format(len(df_ag.query('top_1_rerank_match == True')), len(df_ag)))
    logging.info('---------')

  logging.info('[TRAIN] Reranked Granularity Macro Average: {:.2f}'.format(granularity_score_sum_train*100/len(answer_granularities)))
  logging.info('[TEST] Reranked Granularity Macro Average: {:.2f}'.format(granularity_score_sum_test*100/len(answer_granularities)))
  logging.info('---------')

# ====================================================
# Model Training Functions
# ====================================================

def biencoder_training(df_train):
  model = SentenceTransformer(HPARAMS.reranker_model)

  if (HPARAMS.prepend_component_type):
    we_model = model._first_module()
    we_model.tokenizer.add_tokens(['[GRANULARITY]'], special_tokens=True)
    we_model.auto_model.resize_token_embeddings(len(we_model.tokenizer))

  train_examples = []
  lcs = []
  count_negs = []
  for _,row in df_train.iterrows():
    count_neg = 0
    lc = None
    lc = LawComponent.from_answer_granularity_row(row)
    try:
      _, neg_examples = find_negative_examples(row)
      if HPARAMS.rerank_relevance_only:
        neg_examples = sample_random_negative_examples(lc, HPARAMS.k_random_negative_examples)
      else:
        neg_examples += sample_random_negative_examples(lc, HPARAMS.k_random_negative_examples)
      count_neg = len(neg_examples)
      for neg_example in neg_examples:
        example = {
            'query': question_to_statement(row['Question']),
            'pos': row['text'],
            'neg': neg_example
        }
        if (type(example['query']) == str and type(example['pos']) == str and type(example['neg']) == str):
          train_examples.append(example)
    except ValueError:
      pass
    count_negs.append(count_neg)
    lcs.append(lc)
  df_train['lc'] = lcs

  if (HPARAMS.max_siamese_train_size > 0):
    train_examples = random.sample(train_examples, HPARAMS.max_siamese_train_size)
  logging.info("Train examples: " + str(len(train_examples)))

  train_input_examples = []
  for example in train_examples:
    train_input_examples.append(InputExample(texts=[example['query'], example['pos'], example['neg']]))

  train_dataloader = DataLoader(
      train_input_examples, 
      shuffle=True, 
      batch_size=HPARAMS.batch_size
  )

  train_loss = losses.TripletLoss(model=model)
  
  model.fit(train_objectives=[(train_dataloader, train_loss)], 
            optimizer_params={
                   'lr': HPARAMS.learning_rate
               },
            epochs=HPARAMS.epochs) 

  return model, df_train

def crossencoder_training(df_train):
  ce_model = CrossEncoder(HPARAMS.reranker_model)
  ce_model.max_length = 512

  if (HPARAMS.prepend_component_type):
    ce_model.tokenizer.add_tokens(['[GRANULARITY]'], special_tokens=True)
    ce_model.model.resize_token_embeddings(len(ce_model.tokenizer))

  train_ce_examples = []
  lcs = []
  count_negs = []
  for _,row in df_train.iterrows():
    count_neg = 0
    lc = None
    lc = LawComponent.from_answer_granularity_row(row)
    try:
      _, neg_examples = find_negative_examples(row)
      neg_examples_random = sample_random_negative_examples(lc, HPARAMS.k_random_negative_examples)
      count_neg = len(neg_examples)
      if (type(question_to_statement(row['Question'])) == str and type(row['text']) == str):
        train_ce_examples.append(InputExample(texts=[question_to_statement(row['Question']), row['text']], label=1.0))
      if not HPARAMS.rerank_relevance_only:
        for neg_example in neg_examples:
          example = {
                'query': question_to_statement(row['Question']),
                'passage': neg_example,
                'label': 0.0
            }
          if (type(example['query']) == str and type(example['passage']) == str):
            train_ce_examples.append(InputExample(texts=[example['query'], example['passage']], label=example['label']))
      for neg_example in neg_examples_random:
        example = {
            'query': question_to_statement(row['Question']),
            'passage': neg_example,
            'label': 0.0
        }
        if (type(example['query']) == str and type(example['passage']) == str):
          train_ce_examples.append(InputExample(texts=[example['query'], example['passage']], label=example['label']))
    except ValueError:
      pass
    count_negs.append(count_neg)
    lcs.append(lc)
  df_train['lc'] = lcs

  train_dataloader = DataLoader(
      train_ce_examples, 
      shuffle=True, 
      batch_size=HPARAMS.batch_size
  )

  ce_model.fit(train_dataloader=train_dataloader,
               optimizer_params={
                   'lr': HPARAMS.learning_rate
               },
            epochs=HPARAMS.epochs)

  return ce_model, df_train

# ====================================================
# Model Inference Functions
# ====================================================

def biencoder_inference(model, df):
  top_k_test_results = []

  k = HPARAMS.top_k

  for _,row in df.iterrows():
    question = question_to_statement(row['Question'])
    query = CFG.base_query.copy()
    query['query']['bool']['should'][0]['match']['text']['query'] = question
    query['query']['bool']['should'][1]['match']['chapterTitle']['query'] = question
    query['query']['bool']['should'][2]['match_phrase']['text']['query'] = question
    query['query']['bool']['should'][3]['match_phrase']['chapterTitle']['query'] = question
    
    # keywords & law filter
    keywords = get_keywords_from_query(question)
    if len(keywords) > 0 and HPARAMS.use_keyword_law_filter:
        law_uris = set().union(*[reverse_keywords_dict[k] for k in keywords])
        query['query']['bool']['filter'] = {
            "terms": {
              "lawUri": list(law_uris)
            }
          }
    
    result = es.search(
        index="index_redundant" + "_" + str(cur_timestamp), 
      query=query['query']
    )

    try:
      # top_1_test_results.append(result['hits']['hits'][0])
      top_k_test_results.append(result['hits']['hits'][:k])
    except IndexError:
      # top_1_test_results.append({})
      top_k_test_results.append({})

  # test_data_redundant['top_1'] = top_1_test_results
  df['top_k'] = top_k_test_results

  top_1_rerank_result = []
  top_1_rerank_match = []
  top_1_rerank_article_match = []
  answer_in_top_k = []
  f1_scores = []

  for _,row in df.iterrows():
    expected = LawComponent.from_answer_granularity_row(row)
    result = None
    uri_result = 'wrong'
    answer_in_top_k_i = False
    f1_score = 0
    try:
      query = question_to_statement(row['Question'])
      uris = [i['_source']['uri'] for i in row['top_k']]
      passages = [i['_source']['text'] for i in row['top_k']]
      query_embedding = model.encode(query)
      passage_embedding = model.encode(passages)
      index = np.argmax(util.dot_score(query_embedding, passage_embedding))
      result = passages[index]
      uri_result = uris[index]
      actual_top_1 = LawComponent.from_uri(uri_result)

      actual = row['text']
      f1_score = (compute_qa_f1(actual, result))

      for uri in uris:
        actual = LawComponent.from_uri(uri)
        if expected == actual:
          answer_in_top_k_i = True

    except:
      pass
    top_1_rerank_result.append(result)
    top_1_rerank_match.append(expected == actual_top_1)
    top_1_rerank_article_match.append(expected.is_article_equal(actual_top_1))
    answer_in_top_k.append(answer_in_top_k_i)
    f1_scores.append(f1_score)

  df['top_1_rerank_result'] = top_1_rerank_result
  df['top_1_rerank_match'] = top_1_rerank_match
  df['top_1_rerank_article_match'] = top_1_rerank_article_match
  df['answer_in_top_k'] = answer_in_top_k
  df['f1_score'] = f1_scores

  return df

def crossencoder_inference(model, df):
  top_k_test_results = []

  k = HPARAMS.top_k

  for _,row in df.iterrows():
    question = question_to_statement(row['Question'])
    query = CFG.base_query.copy()
    query['query']['bool']['should'][0]['match']['text']['query'] = question
    query['query']['bool']['should'][1]['match']['chapterTitle']['query'] = question
    query['query']['bool']['should'][2]['match_phrase']['text']['query'] = question
    query['query']['bool']['should'][3]['match_phrase']['chapterTitle']['query'] = question
    
    keywords = get_keywords_from_query(question)
    if len(keywords) > 0 and HPARAMS.use_keyword_law_filter:
        law_uris = set().union(*[reverse_keywords_dict[k] for k in keywords])
        query['query']['bool']['filter'] = {
            "terms": {
              "lawUri": list(law_uris)
            }
          }
    
    result = es.search(
        index="index_redundant" + "_" + str(cur_timestamp), 
        query=query['query']
    )

    try:
      # top_1_test_results.append(result['hits']['hits'][0])
      top_k_test_results.append(result['hits']['hits'][:k])
    except IndexError:
      # top_1_test_results.append({})
      top_k_test_results.append({})

  # test_data_redundant['top_1'] = top_1_test_results
  df['top_k'] = top_k_test_results

  top_1_rerank_result = []
  top_1_rerank_match = []
  top_1_rerank_article_match = []
  answer_in_top_k = []
  f1_scores = []

  for _,row in df.iterrows():
    expected = LawComponent.from_answer_granularity_row(row)
    result = None
    uri_result = 'wrong'
    answer_in_top_k_i = False
    f1_score = 0
    try:
      query = question_to_statement(row['Question'])
      uris = [i['_source']['uri'] for i in row['top_k']]
      passages = [i['_source']['text'] for i in row['top_k']]
      sentence_combinations = [[query, corpus_sentence] for corpus_sentence in passages]
      similarity_scores = model.predict(sentence_combinations)
      index = np.argmax(similarity_scores)
      result = passages[index]
      uri_result = uris[index]
      actual_top_1 = LawComponent.from_uri(uri_result)

      actual = row['text']
      f1_score = (compute_qa_f1(actual, result))

      for uri in uris:
        actual = LawComponent.from_uri(uri)
        if expected == actual:
          answer_in_top_k_i = True

    except:
      pass
    top_1_rerank_result.append(result)
    top_1_rerank_match.append(expected == actual_top_1)
    top_1_rerank_article_match.append(expected.is_article_equal(actual_top_1))
    answer_in_top_k.append(answer_in_top_k_i)
    f1_scores.append(f1_score)

  df['top_1_rerank_result'] = top_1_rerank_result
  df['top_1_rerank_match'] = top_1_rerank_match
  df['top_1_rerank_article_match'] = top_1_rerank_article_match
  df['answer_in_top_k'] = answer_in_top_k
  df['f1_score'] = f1_scores

# ========================================================================================================
# Dataset Preprocessing
# ========================================================================================================
# ====================================================
# Dataset Preprocessing for Elastic Search
# ====================================================

df = pd.read_csv(CFG.BASE_DIR + '2023-04-20_pp-dataset.csv', dtype=CFG.col_types)
df = df.drop_duplicates(subset=['uri']).reset_index()
df['uri'] = df['uri'].apply(lambda x: x.replace('http://', 'https://'))
for col in CFG.fill_values_dataset:
  df[col] = df[col].fillna(CFG.fill_values_dataset[col])

df['law_component'] = df['uri'].apply(LawComponent.from_uri)
df['law_component_str'] = df['uri'].apply(lambda uri: str(LawComponent.from_uri(uri)))

logging.info("Law Components counter: " + str(Counter(df['type'].to_list())))

df['keywords'] = df.query('nomorBab == 1 and type == "huruf-pasal"')['isiTeks'].apply(extract_definition_keyword)

keywords_dict = {}

for law_uri in df['peraturan'].unique():
  definition_keywords_list = df.query('peraturan == @law_uri and not keywords.isnull()', engine='python')['keywords'].to_list()
  definition_keywords_set = set().union(*definition_keywords_list)
  keywords_dict[law_uri] = definition_keywords_set

reverse_keywords_dict = {}

for law_uri, keywords in keywords_dict.items():
  for keyword in keywords:
    if keyword in reverse_keywords_dict:
      reverse_keywords_dict[keyword].append(law_uri)
    else:
      reverse_keywords_dict[keyword] = [law_uri]

# ====================================================
# Dataset Preprocessing for QA
# ====================================================

test_data = pd.read_csv(CFG.BASE_DIR + 'Granularity-Based Law QA Dataset - 2023-04-20.csv', dtype=CFG.test_data_types)

test_data['Answer Granularity'] = test_data['Answer Granularity'].replace(CFG.answer_granularity_dict)
for col in CFG.fill_values_qa_dataset:
  test_data[col] = test_data[col].fillna(CFG.fill_values_qa_dataset[col])

test_data_redundant = test_data.copy(deep=True)

# ========================================================================================================
# Retrieval using BM25
# ========================================================================================================
# ====================================================
# Initialize Elastic Search
# ====================================================

es = Elasticsearch(hosts=[CFG.ELASTICSEARCH_HOST])
es.indices.delete(index='index_redundant*', ignore=[400, 404]) # for overlapping index
es_logger = logging.getLogger('elasticsearch')
es_logger.setLevel(logging.WARNING)
cur_timestamp = int(time.time())

filter_temp = ["lowercase"]

if HPARAMS.use_stop_only:
    filter_temp.append("stop_id")
elif HPARAMS.use_stem_only:
    filter_temp.append("stemmer_id")

mapping = {
    "settings": {
        "analysis": {
          "analyzer": {
            "index_analyzer": {
              "tokenizer": "standard",
              "filter": filter_temp
            },
              "stop_stemmer_analyzer": {
                  "tokenizer": "standard",
                  "filter": [ 
                      "lowercase",
                      "stop_id", 
                      "stemmer_id" 
                  ]
              },
            "synonym_stop_analyzer": {
              "tokenizer": "standard",
              "filter": [ 
                  "lowercase",
              ]
            },
            "label_tokenizer": {
                "tokenizer": "whitespace"
            }
          },
          "filter": {
            "stop_id": {
                "type": "stop",
                "stopwords": "_indonesian_"
            },
            "synonym": {
              "type": "synonym",
              "synonyms": [
                  "phk, pemecatan, penghentian kerja, putus kontrak => pemutusan hubungan kerja",
                  "mangkir, bolos, tidak masuk kerja, strike => mogok kerja",
                  "pkwt => perjanjian kerja waktu tertentu",
                  "karyawan, pegawai => pekerja",
                  "kontrak kerja => perjanjian kerja",
                  "pkwtt => perjanjian kerja waktu tidak tertentu",
                  "resign => pengunduran diri"
              ]
            },
            "stemmer_id": {
                "type": "stemmer",
                "language": "indonesian"
            }
          }
      }
    },
    "mappings": {
        "properties": {
            "text": {
                "type": "text", # formerly "string"
                "analyzer": "index_analyzer" if not HPARAMS.use_stop_removal_stemming else "stop_stemmer_analyzer",
                "search_analyzer": "index_analyzer" if not HPARAMS.use_stop_removal_stemming else "stop_stemmer_analyzer"
            },
            "chapterTitle": {
                "type": "text",
                "analyzer": "index_analyzer" if not HPARAMS.use_stop_removal_stemming else "stop_stemmer_analyzer",
                "search_analyzer": "index_analyzer" if not HPARAMS.use_stop_removal_stemming else "stop_stemmer_analyzer"
            },
            "uri": {
                "type": "text"
            },
            "lawUri": {
                "type": "keyword"
            }
        }
    }
}

# make an API call to the Elasticsearch cluster and have it return a response:
response = es.indices.create(
    index="index_redundant" + "_" + str(cur_timestamp),
    body=mapping,
    ignore=400 # ignore 400 already exists code
)

time.sleep(3) # short wait after creating index

# ====================================================
# Indexing Elastic Search
# ====================================================

for _,row in df.iterrows():
  text = row['isiTeks']
  if (not pd.isnull(row['description'])):
    text = row['description'] + ' ' + text
  if (not pd.isnull(row['judulBab'])):
    chapterTitle = row['judulBab']
  else:
    chapterTitle = ""
  data = {
      'uri': row['uri'],
      'text': text,
      'type': row['type'],
      'chapterTitle': chapterTitle,
      'lawUri': row['peraturan']
  }
  if (HPARAMS.article_only_index and row['type'] != 'pasal'):
    continue
  if (HPARAMS.chapter_title_only_for_chapter and row['type'] != 'bab'):
    data['chapterTitle'] = ''
  es.index(
      index="index_redundant" + "_" + str(cur_timestamp), 
      id=str(uuid.uuid4()), 
      document=json.dumps(data)
  )
    
time.sleep(30) # short sleep after indexing

# ====================================================
# Query Elastic Search
# ====================================================

if (HPARAMS.include_map):
    in_line_indexes_list = []

    for _,row in test_data_redundant.iterrows():
        in_line_indexes = find_in_line_indexes(row)
        in_line_indexes_list.append(in_line_indexes)

    test_data_redundant['in_line_indexes'] = in_line_indexes_list

top_1_test_results = []
top_k_test_results = []
ap_results = []

for _,row in test_data_redundant.iterrows():
  top_1_test_result, top_k_test_result, ap_result = get_test_result(row)
  top_1_test_results.append(top_1_test_result)
  top_k_test_results.append(top_k_test_result)
  if HPARAMS.include_map:
    ap_results.append(ap_result)

test_data_redundant['top_1'] = top_1_test_results
test_data_redundant['top_k'] = top_k_test_results

if HPARAMS.include_map:
    test_data_redundant['ap'] = ap_results

# ====================================================
# Evaluate Retriever
# ====================================================

top_1_article_correct = []
top_1_all_correct = []
top_1_all_class_correct = []
answer_in_top_k_list = []
f1_scores = []

expected_text = []
for i in range(len(test_data_redundant)):
  expected_text.append(get_text(test_data_redundant.iloc[i]))
test_data_redundant['expected_text'] = expected_text

for _,row in test_data_redundant.iterrows():
  metric_scores = evaluate_retriever(row)
  top_1_article_correct.append(metric_scores["article_correct"])
  top_1_all_correct.append(metric_scores["all_correct"])
  answer_in_top_k_list.append(metric_scores["answer_in_top_k"])
  f1_scores.append(metric_scores["f1_score"])
      
test_data_redundant['top_1_article_correct'] = top_1_article_correct
test_data_redundant['top_1_all_correct'] = top_1_all_correct
test_data_redundant['answer_in_top_k'] = answer_in_top_k_list
test_data_redundant['f1_score'] = f1_scores

article_correct_acc = len(test_data_redundant.query('top_1_article_correct == True')) / len(test_data_redundant.query('`Answer Granularity` != "Chapter"'))
all_correct_acc = len(test_data_redundant.query('top_1_all_correct == True')) / len(test_data_redundant)
answer_in_top_k_score = len(test_data_redundant.query('answer_in_top_k == True')) / len(test_data_redundant)
qa_f1_score = test_data_redundant['f1_score'].mean()

logging.info('QA F1 Score Average {:.2f}'.format(qa_f1_score*100))
logging.info('Article Match {:.2f}'.format(article_correct_acc*100))
logging.info('Exact Match {:.2f}'.format(all_correct_acc*100))
logging.info('Percent Answer in Top-k {:.2f}'.format(answer_in_top_k_score*100))

if HPARAMS.include_map:
    logging.info('mAP {:.2f}'.format(test_data_redundant['ap'].mean()*100))

# ====================================================
# Baseline Result Per Category
# ====================================================

answer_granularities = test_data_redundant['Answer Granularity'].unique()

logging.info('Baseline Result Per Category')
logging.info('============================')

granularity_score_sum = 0

for i in answer_granularities:

  df_ag = test_data_redundant.query('`Answer Granularity` == @i')

  try:
    article_correct_acc = len(df_ag.query('top_1_article_correct == True')) / len(df_ag.query('`Answer Granularity` != "Chapter"'))
  except:
    article_correct_acc = 0
  all_correct_acc = len(df_ag.query('top_1_all_correct == True')) / len(df_ag)
  answer_in_top_k_score = len(df_ag.query('answer_in_top_k == True')) / len(df_ag)
  qa_f1_score = df_ag['f1_score'].mean()
  granularity_score_sum += all_correct_acc


  logging.info('Answer Granularity: {}'.format(i))
  logging.info('QA F1 Score Average {:.2f}'.format(qa_f1_score*100))
  logging.info('Article Match {:.2f}'.format(article_correct_acc*100))
  logging.info('Exact Match {:.2f}'.format(all_correct_acc*100))
  if HPARAMS.include_map:
    logging.info('mAP {:.2f}'.format(df_ag['ap'].mean()*100))
  logging.info('Percent Answer in Top-k {:.2f}'.format(answer_in_top_k_score*100))
  logging.info('---------')

logging.info('Baseline Granularity Macro Average: {:.2f}'.format(granularity_score_sum*100/len(answer_granularities)))
logging.info('---------')    

# ====================================================
# Retrieval Result Per Category
# ====================================================

# also used for reranker
df_train, df_test = train_test_split_from_retriever(test_data_redundant)

logging.info("Siamese Train Data answers:\n" + str(Counter(df_train['Answer Granularity'].to_list())))
logging.info("Siamese Test Data answers:\n" + str(Counter(df_test['Answer Granularity'].to_list())))

text = []
for i in range(len(df_train)):
  text.append(get_text(df_train.iloc[i]))
df_train['text'] = text

text = []
for i in range(len(df_test)):
  text.append(get_text(df_test.iloc[i]))
df_test['text'] = text

article_correct_acc = len(df_train.query('top_1_article_correct == True')) / len(df_train.query('`Answer Granularity` != "Chapter"'))
all_correct_acc = len(df_train.query('top_1_all_correct == True')) / len(df_train)
answer_in_top_k_score = len(df_train.query('answer_in_top_k == True')) / len(df_train)
qa_f1_score = df_train['f1_score'].mean()

logging.info('[TRAIN] Retrieval QA F1 Score Average {:.2f}'.format(qa_f1_score*100))
logging.info('[TRAIN] Retrieval Article Match {:.2f}'.format(article_correct_acc*100))
logging.info('[TRAIN] Retrieval Exact Match {:.2f}'.format(all_correct_acc*100))
logging.info('[TRAIN] Retrieval Percent Answer in Top-k {:.2f}'.format(answer_in_top_k_score*100))

if HPARAMS.include_map:
    logging.info('[TRAIN] Retrieval mAP {:.2f}'.format(df_train['ap'].mean()*100))

article_correct_acc = len(df_test.query('top_1_article_correct == True')) / len(df_test.query('`Answer Granularity` != "Chapter"'))
all_correct_acc = len(df_test.query('top_1_all_correct == True')) / len(df_test)
answer_in_top_k_score = len(df_test.query('answer_in_top_k == True')) / len(df_test)
qa_f1_score = df_test['f1_score'].mean()

logging.info('[TEST] Retrieval QA F1 Score Average {:.2f}'.format(qa_f1_score*100))
logging.info('[TEST] Retrieval Article Match {:.2f}'.format(article_correct_acc*100))
logging.info('[TEST] Retrieval Exact Match {:.2f}'.format(all_correct_acc*100))
logging.info('[TEST] Retrieval Percent Answer in Top-k {:.2f}'.format(answer_in_top_k_score*100))

if HPARAMS.include_map:
    logging.info('[TEST] Retrieval mAP {:.2f}'.format(df_test['ap'].mean()*100))

answer_granularities = test_data_redundant['Answer Granularity'].unique()

logging.info('Retrieval Result Per Category')
logging.info('============================')

granularity_score_sum_train = 0
granularity_score_sum_test = 0

for i in answer_granularities:

    df_ag = df_train.query('`Answer Granularity` == @i')

    all_correct_acc = len(df_ag.query('top_1_all_correct == True')) / len(df_ag)
    article_correct_acc = len(df_ag.query('top_1_article_correct == True')) / len(df_ag)
    granularity_score_sum_train += all_correct_acc

    if (HPARAMS.log_train_results):
        logging.info('[TRAIN-Retrieval] Answer Granularity: {}'.format(i))
        logging.info('[TRAIN-Retrieval] QA F1 Score Average: {:.2f}'.format(df_ag['f1_score'].mean() * 100))
        logging.info('[TRAIN-Retrieval] Article Match {:.2f}'.format(article_correct_acc*100))
        logging.info('[TRAIN-Retrieval] Exact Match {:.2f}'.format(all_correct_acc*100))
        if HPARAMS.include_map:
            logging.info('[TRAIN-Retrieval] Retrieval mAP {:.2f}'.format(df_ag['ap'].mean()*100))
        logging.info('[TRAIN-Retrieval] Test Count: ({}/{})'.format(len(df_ag.query('top_1_all_correct == True')), len(df_ag)))
        logging.info('---------')

    df_ag = df_test.query('`Answer Granularity` == @i')

    all_correct_acc = len(df_ag.query('top_1_all_correct == True')) / len(df_ag)
    article_correct_acc = len(df_ag.query('top_1_article_correct == True')) / len(df_ag)
    granularity_score_sum_test += all_correct_acc

    logging.info('[TEST-Retrieval] Answer Granularity: {}'.format(i))
    logging.info('[TEST-Retrieval] QA F1 Score Average: {:.2f}'.format(df_ag['f1_score'].mean() * 100))
    logging.info('[TEST-Retrieval] Article Match {:.2f}'.format(article_correct_acc*100))
    logging.info('[TEST-Retrieval] Exact Match {:.2f}'.format(all_correct_acc*100))
    if HPARAMS.include_map:
        logging.info('[TEST-Retrieval] Retrieval mAP {:.2f}'.format(df_ag['ap'].mean()*100))
    logging.info('[TEST-Retrieval] Test Count: ({}/{})'.format(len(df_ag.query('top_1_all_correct == True')), len(df_ag)))
    logging.info('---------')

logging.info('[TRAIN] Retrieval Granularity Macro Average: {:.2f}'.format(granularity_score_sum_train*100/len(answer_granularities)))
logging.info('[TEST] Retrieval Granularity Macro Average: {:.2f}'.format(granularity_score_sum_test*100/len(answer_granularities)))
logging.info('---------')

# ========================================================================================================
# Reranker using BiEncoder or CrossEncoder
# ========================================================================================================

if (HPARAMS.reranker_mode == "BiEncoder"):
  model, df_train = biencoder_training(df_train)
  df_train = biencoder_inference(model, df_train)
  df_test = biencoder_inference(model, df_test)
elif (HPARAMS.reranker_mode == "CrossEncoder"):
  model, df_train = crossencoder_training(df_train)
  crossencoder_inference(model, df_train)
  crossencoder_inference(model, df_test)
log_model_results(df_train, df_test)

logging.info("END Experiment")
logging.shutdown()
torch.cuda.empty_cache()
