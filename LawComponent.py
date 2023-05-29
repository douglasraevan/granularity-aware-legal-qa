import re
import copy
import pandas as pd

LAW_PATTERN = '(?P<law_type>[A-Za-z]+) (?P<law_number>\d{1,})/(?P<law_year>\d{4})'

class LawComponent:

  law_type: str = None
  law_year: int = 0
  law_number: int = 0
  component_type: str = None
  chapter: int = None
  article: str = None
  subsection: int = None
  letter: str = None

  def __init__(self):
    pass

  def from_uri(uri: str):
    lc = LawComponent()
    uri_split = uri.split('/')
    lc.law_type = uri_split[4]
    lc.law_year = int(uri_split[5]) if int(uri_split[5]) != 0 else None
    lc.law_number = int(uri_split[6]) if int(uri_split[6]) != 0 else None
    if len(uri_split) < 8:
      return lc
    if (uri_split[7] == 'bab'):
      lc.component_type = 'chapter'
      lc.chapter = int(uri_split[8]) if int(uri_split[8]) != 0 else None
    else:
      lc.article = str(int(uri_split[8])) if int(uri_split[8]) != 0 else None
      lc.component_type = 'article'
      if (len(uri_split) > 9 and uri_split[9] == "versi"):
        if (len(uri_split) > 11 and uri_split[11] == "ayat"):
          lc.subsection = int(uri_split[12]) if int(uri_split[12]) != 0 else None
          lc.component_type = 'subsection'
          if (len(uri_split) > 13 and uri_split[13] == "huruf"):
            lc.component_type = 'letter'
            try:
              lc.letter = str(int(uri_split[14])) if int(uri_split[14]) != 0 else None
            except:
              lc.letter = uri_split[14]
        elif (len(uri_split) > 11 and uri_split[11] == "huruf"):
          lc.component_type = 'letter'
          try:
            lc.letter = str(int(uri_split[12])) if int(uri_split[12]) != 0 else None
          except:
            lc.letter = uri_split[12]

    return lc

  def from_answer_granularity_row(row_dict: dict):
    lc = LawComponent()

    law = row_dict['Law']
    law_search = re.search(LAW_PATTERN, law)

    if (law_search != None):
      lc.law_type = law_search.group('law_type').lower()
      lc.law_number = int(law_search.group('law_number')) if int(law_search.group('law_number')) != 0 else None
      lc.law_year = int(law_search.group('law_year')) if int(law_search.group('law_year')) != 0 else None

      lc.component_type = row_dict['Answer Granularity'].lower()
      lc.chapter = int(row_dict['Chapter']) if int(row_dict['Chapter']) != 0 else None
      try:
        lc.article = str(int(row_dict['Article'])) if int(row_dict['Article']) != 0 else None
      except:
        pass
      try:
        lc.subsection = int(row_dict['Subsection']) if int(row_dict['Subsection']) != 0 else None
      except:
        pass
      try:
        lc.letter = str(int(row_dict['Letter (1st level)'])) if int(row_dict['Letter (1st level)']) != 0 else None
      except:
        if (pd.isnull(row_dict['Letter (1st level)'])):
          lc.letter = None
        else:
          lc.letter = row_dict['Letter (1st level)']
      
    return lc

  def __eq__(self, other):

    if not (self.law_type == other.law_type and self.law_year == other.law_year 
            and self.law_number == other.law_number):
      return False

    if self.component_type != other.component_type:
      return False

    # if self.component_type == 'chapter':
    #   if self.chapter != other.chapter:
    #     return False
    if self.article == None and other.article == None:
      if self.chapter != other.chapter:
        return False
    else:
      if self.article != other.article:
        return False

      if self.component_type == 'article':
        return True
      
      if self.subsection != other.subsection:
        return False

      if self.component_type == 'subsection':
        return True

      if self.letter != other.letter:
        return False

      if self.component_type == 'letter':
        return True

    return True


  def is_article_equal(self, other):

    if not (self.law_type == other.law_type and self.law_year == other.law_year 
            and self.law_number == other.law_number):
      return False

    if self.component_type == 'chapter':
      return False
    else:
      if self.article != other.article:
        return False

    return True

  def __repr__(self):
    return "LawComponent({}, {}, {}, {}, {}, {}, {}, {})".format(
        self.law_type,
        self.law_number,
        self.law_year,
        self.component_type,
        self.chapter,
        self.article,
        self.subsection,
        self.letter
    )
  
  def __str__(self):
    return self.__repr__()

  def copy(other):
    return copy.deepcopy(other)