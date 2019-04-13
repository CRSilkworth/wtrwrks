import transform as n
import pandas as pd
import numpy as np
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
# import production.nlp.ja_stopwords as jas
# import production.nlp.ko_stopwords as kos
# import production.nlp.zh_hans_stopwords as zss
import unicodedata
import tinysegmenter
from chop.hmm import Tokenizer as HMMTokenizer
from chop.mmseg import Tokenizer as MMSEGTokenizer
import operator
import konlpy

def get_wordnet_pos(treebank_tag):
  if treebank_tag.startswith('J'):
    return wordnet.ADJ
  elif treebank_tag.startswith('V'):
    return wordnet.VERB
  elif treebank_tag.startswith('N'):
    return wordnet.NOUN
  elif treebank_tag.startswith('R'):
    return wordnet.ADV
  else:
    return wordnet.NOUN


class StringTransform(n.Transform):
  """Class used to create mappings from a raw string to a vector and back.

  Parameters
  ----------
  df : pd.DataFrame
    The dataframe with all the data used to define the mappings.
  columns : list of strs
    The column names of all the relevant columns that make up the data to be taken from the dataframe
  language: str
    The language the strings are in.
  lower: bool
    Whether or not to lowercase all the strings.
  half_width: bool
    Whether or not to convert all full width characters to half width.
  lemmatize: bool
    Whether or not to stem or lemmatize the words.
  remove_stopwords: bool
    Whether or not to remove all the commonly occuring words.
  max_vocab_size:
    The maximum number of words allowed to be a part of the known vocabulary.
  from_file : str
    The path to the saved file to recreate the transform object that was saved to disk.
  save_dict : dict
    The dictionary to rereate the transform object

  Attributes
  ----------
  attribute_list : list of strs
    The list of attributes that need to be saved in order to fully reconstruct the transform object.

  """

  attribute_list = ['index_to_word', 'word_to_index', 'columns', 'key', 'language', 'lower', 'half_width', 'lemmatize', 'remove_stopwords', 'normalize_whitespace', 'max_vocab_size', 'vector_size']

  def _setattributes(self, df, columns, language, lower=False, half_width=False, lemmatize=False, normalize_whitespace=True, remove_stopwords=False, vector_size=10, max_vocab_size=10000, **kwargs):
    """Set all the relevant attributes for this subclass. Called by the constructor for the Transform class.

    Parameters
    ----------
    df : pd.DataFrame
      The dataframe with all the data used to define the mappings.
    columns : list of strs
      The column names of all the relevant columns that make up the data to be taken from the dataframe
    language: str
      The language the strings are in.
    lower: bool
      Whether or not to lowercase all the strings.
    half_width: bool
      Whether or not to convert all full width characters to half width.
    lemmatize: bool
      Whether or not to stem or lemmatize the words.
    remove_stopwords: bool
      Whether or not to remove all the commonly occuring words.
    max_vocab_size:
      The maximum number of words allowed to be a part of the known vocabulary.
    """
    self.columns = columns
    self.language = language
    self.lower = lower
    self.half_width = half_width
    self.lemmatize = lemmatize
    self.remove_stopwords = remove_stopwords
    self.max_vocab_size = max_vocab_size
    self.vector_size = vector_size
    self.normalize_whitespace = normalize_whitespace

    # Set some various attributes used to normalize the string
    if lemmatize:
      self.lemmatizer = WordNetLemmatizer()
    if remove_stopwords:
      self.stopwords = set(nltk.corpus.stopwords.words('english'))
    if language == 'zh_hans':
      self.tokenizer = HMMTokenizer()
    elif language == 'zh_hant':
      self.tokenizer = MMSEGTokenizer()
    elif language == 'ja':
      self.tokenizer = tinysegmenter.TinySegmenter()
    elif language == 'ko':
      self.tokenizer = konlpy.tag.Kkma()

    # Get all the words and the number of times they appear
    all_words = {}
    for column_name in columns:
      for string in df[column_name]:
        # Tokenize each request, add the tokens to the set of all words
        tokens = self._normalize_string(string)
        for token_num, token in enumerate(tokens):
          all_words.setdefault(token, 0)
          all_words[token] += 1

    # Sort the dict by the number of times the words appear
    sorted_words = sorted(all_words.items(), key=operator.itemgetter(1), reverse=True)

    # Pull out the first 'max_vocab_size' words
    sorted_words = [w for w, c in sorted_words[:max_vocab_size - 1]]

    # Create the mapping from category values to index in the vector and
    # vice versa
    self.index_to_word = sorted(sorted_words)
    self.index_to_word = ['__UNK__'] + self.index_to_word
    self.word_to_index = {
      word: num for num, word in enumerate(self.index_to_word)
    }

  def _normalize_string(self, string):
    """Tokenize, lemmatize, maybe strip out stop words, maybe lemmatize a string in a consistent manner.

    Parameters
    ----------
    string : str
      The string to be tokenized and normalized

    Returns
    -------
    list of strs
      The tokenized and normalized strings

    """
    if self.normalize_whitespace:
      string = ' '.join(string.split())
    if self.language == 'en':
      string = self._en_normalize(string)
    elif self.language == 'ja':
      string = self._ja_normalize(string)
    elif self.language == 'ko':
      string = self._ko_normalize(string)
    elif self.language == 'zh_hans':
      string = self._zh_hans_normalize(string)
    elif self.language == 'zh_hant':
      string = self._zh_hant_normalize(string)

    return string

  def _en_normalize(self, string):
    """Tokenize, lemmatize, maybe strip out stop words, maybe lemmatize an English string in a consistent manner.

    Parameters
    ----------
    string : str
      The string to be tokenized and normalized

    Returns
    -------
    list of strs
      The tokenized and normalized strings

    """
    r_tokens = []

    # Split the string into individual words/punctuation.
    tokens = nltk.word_tokenize(string)

    # If lemmatize, then stem the words according to their part of speech.
    if self.lemmatize:
      pos_tags = nltk.pos_tag(tokens)
    for token_num, token in enumerate(tokens):

      # If you're lower casing everything then do it here.
      if self.lower:
        token = token.lower()

      # If you're removing stop words and this is a stop word continue.
      if self.remove_stopwords and token in self.stopwords:
        continue

      # If lemmatize, then lemmatize it
      if self.lemmatize:
        pos_tag = get_wordnet_pos(pos_tags[token_num][1])
        token = self.lemmatizer.lemmatize(token, pos_tag)

      r_tokens.append(token)
    return r_tokens

  def _ja_normalize(self, string):
    """Tokenize, lemmatize, maybe strip out stop words, maybe lemmatize a Japanese string in a consistent manner.

    Parameters
    ----------
    string : str
      The string to be tokenized and normalized

    Returns
    -------
    list of strs
      The tokenized and normalized strings

    """
    r_tokens = []

    # If half_width then convert all full width characters to half width
    if self.half_width:
      string = unicodedata.normalize('NFKC', unicode(string))

    # Split the string into individual words/punctuation. Iterate through.
    for token in self.tokenizer.tokenize(string):

      # If you're removing stop words and this is a stop word continue.
      if self.remove_stopwords and token in jas.stopwords:
        continue
      r_tokens.append(token)
    return r_tokens

  def _ko_normalize(self, string):
    """Tokenize, lemmatize, maybe strip out stop words, maybe lemmatize a Korean string in a consistent manner.

    Parameters
    ----------
    string : str
      The string to be tokenized and normalized

    Returns
    -------
    list of strs
      The tokenized and normalized strings

    """
    r_tokens = []

    # If half_width then convert all full width characters to half width
    if self.half_width:
      string = unicodedata.normalize('NFKC', unicode(string))

    # konlpy messes up try except blocks for some reason. Only load when
    # needed.
    # if not hasattr(self, 'tokenizer'):
    #   self.tokenizer = konlpy.tag.Kkma()

    # Split the string into individual words/punctuation. Iterate through.
    # for token, pos in self.tokenizer.pos(string):
    for token in string.split():
      # If you're removing stop words and this is a stop word continue.
      if self.remove_stopwords and token in kos.stopwords:
        continue
      r_tokens.append(token)

    return r_tokens

  def _zh_hans_normalize(self, string):
    """Tokenize, lemmatize, maybe strip out stop words, maybe lemmatize a simplified Chinese string in a consistent manner.

    Parameters
    ----------
    string : str
      The string to be tokenized and normalized

    Returns
    -------
    list of strs
      The tokenized and normalized strings

    """
    r_tokens = []

    # If half_width then convert all full width characters to half width
    if self.half_width:
      string = unicodedata.normalize('NFKC', unicode(string))

    # Split the string into individual words/punctuation. Iterate through.
    for token in self.tokenizer.cut(string):

      # If you're removing stop words and this is a stop word continue.
      if self.remove_stopwords and token in zss.stopwords:
        continue
      r_tokens.append(token)
    return r_tokens

  def _zh_hant_normalize(self, string):
    """Tokenize, lemmatize, maybe strip out stop words, maybe lemmatize a traditional Chinese string in a consistent manner.

    Parameters
    ----------
    string : str
      The string to be tokenized and normalized

    Returns
    -------
    list of strs
      The tokenized and normalized strings

    """
    r_tokens = []

    # If half_width then convert all full width characters to half width
    if self.half_width:
      string = unicodedata.normalize('NFKC', unicode(string))

    # Split the string into individual words/punctuation. Iterate through.
    for token in self.tokenizer.cut(string):

      # If you're removing stop words and this is a stop word continue.
      if self.remove_stopwords and token in zss.stopwords:
        continue
      r_tokens.append(token)
    return r_tokens

  def row_to_vector(self, row, verbose=True):
    """Convert a row in a dataframe to a vector.

    Parameters
    ----------
    row : pd.Series
      A row in a dataframe where the index is the column name and the value is the column value.
    verbose : bool
      Whether or not to print out warnings.

    Returns
    -------
    np.array(
      shape=[len(self)],
      dtype=np.float64
    )
      The vectorized and normalized data.

    """

    # Find the indices for each word, filling with -1 if the vector is
    # longer than the number of tokens.
    indices = -1 * np.ones([self.vector_size], dtype=np.int64)
    token_num = 0
    for column in self.columns:
      string = row[column]
      tokens = self._normalize_string(string)
      for token in tokens:
        # If the max size has been reached then break.
        if token_num == self.vector_size:
          break

        # If the word isn't known, fill it with the 'UNK' index.
        # Otherwise pull out the relevant index.
        if token not in self.word_to_index:
          index = self.word_to_index['__UNK__']
        else:
          index = self.word_to_index[token]

        # Fill the array and increase the token number.
        indices[token_num] = index
        token_num += 1

    return indices

  def vector_to_row(self, vector, verbose=True):
    """Convert the vectorized and normalized data back into it's raw form. Although a lot of information is lost and so it's best to also keep some outside reference to the original row in the dataframe.

    Parameters
    ----------
    vector : np.array(
      shape=[len(self)],
      dtype=np.int64
    )
      The vectorized and normalized data.
    verbose : bool
      Whether or not to print out warnings.

    Returns
    -------
    row : pd.Series
      A row in a dataframe where the index is the column name and the value is the column value.

    """
    # Fill a dictionary representing the original row. Convert the index to
    # a category value
    row = {c: None for c in self.columns}
    first_column = self.columns[0]

    words = []
    for index in vector:
      if index == - 1:
        continue
      words.append(self.index_to_word[index])

    row[first_column] = ' '.join(words)

    # Conver the dict into a pandas series, representing the row.
    return pd.Series([row[c] for c in self.columns], index=self.columns)

  def input_summary(self, key, rows, verbose=True, tabs=0, top_words=20):
    """Create a summary and print out some summary statistics of the the whole dataframe given by rows.

    Parameters
    ----------
    key : str
      A key used to identify the data in tensorboard.
    rows : pd.DataFrame
      The dataframe with all the data to be summarized.
    verbose : bool
      Whether or not to print out the summary statistics.
    tabs : int (default 0)
      Number of tabs to indent the summary statistics.

    Returns
    -------
    tf.Summary
      The summary object used to create the tensorboard histogram.

    """
    # Get the lengths of all the strings
    string_lengths = []
    all_words = {}
    for row in rows:
      string_lengths.append(len(row[0].split(' ')))

      # Count the number of the occurences of each word
      string = row[0]
      tokens = string.split(' ')
      for token in tokens:
        all_words.setdefault(token, 0)
        all_words[token] += 1
    sorted_words = sorted(all_words.items(), key=operator.itemgetter(1), reverse=True)

    max_word = sorted_words[0][1]

    # Create a histogram with each index getting it's own bin.
    lengths_summary = so.summary_histogram(key + '_lengths', string_lengths, bins=np.arange(self.vector_size + 1))
    counts_summary = so.summary_histogram(key + '_word_counts', all_words.values(), bins=np.arange(max_word + 1))

    # Print out some summary statistics
    if verbose:
      print '\t'*tabs, '-'*50
      print '\t'*tabs, key, 'summary info:'
      print '\t'*tabs, '-'*50
      print '\t' * (tabs + 1), 'Vocabulary size:', len(self.word_to_index)

      print
      print '\t' * (tabs + 1), 'Top words by num occurences:'
      for word_num, (word, count) in enumerate(sorted_words):
        if word_num == top_words:
          break
        print '\t' * (tabs + 1), word, count

    return [lengths_summary, counts_summary]

  def _from_save_dict(self, save_dict):
    """Reconstruct the transform object from the dictionary of attributes."""
    for key in self.attribute_list:
      setattr(self, key, save_dict[key])

    if self.language == 'zh_hans':
      self.tokenizer = HMMTokenizer()
    elif self.language == 'zh_hant':
      self.tokenizer = MMSEGTokenizer()
    elif self.language == 'ja':
      self.tokenizer = tinysegmenter.TinySegmenter()
    # konlpy messes up try except blocks for some reason. Only load when
    # needed.
    # elif self.language == 'ko':
    #   self.tokenizer = konlpy.tag.Kkma()

  def __len__(self):
    """Get the length of the vector outputted by the row_to_vector method."""
    return self.vector_size
