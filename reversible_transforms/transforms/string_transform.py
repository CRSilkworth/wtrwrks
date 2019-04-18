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
  supported_languages = {'en': 'English', 'ja': 'Japanese', 'zh_hans': 'Simplified Chinese', 'zh_hant': 'Traditional Chinese', 'ko': 'Korean'}

  attribute_dict = {'col_index': None, 'name': '', 'dtype': np.int64, 'input_dtype': None, 'language': None, 'normalizer': None, 'normalizer_kwargs': None, 'index_to_word': None, 'word_to_index': None, 'max_vocab_size': None, 'max_sent_len': 20, 'tokenizer': None}

  def _setattributes(self, **kwargs):
    super(StringTransform, self)._setattributes(self.attribute_dict, **kwargs)

    if self.norm_mode not in (None, 'mean_std'):
      raise ValueError(self.norm_mode + " not a valid norm mode.")

    if self.index_to_word is None and self.max_vocab_size is None:
      raise ValueError("Must supply a max_vocab_size when, no index_to_word is given.")
    if self.index_to_word is not None and self.index_to_word[0] != '__UNK__':
      raise ValueError("First element of the index_to_word map must be the default '__UNK__' token")

    if self.normalizer is None:
      if self.language in self.supported_languages:
        mod_name = 'reversible_transforms.string_manipulations.' + self.language + '_normalizers'
        mod = __import__(mod_name, globals(), locals(), ['default_normalizer'])
        self.normalizer = getattr(mod, 'default_normalizer')
      else:
        raise ValueError("Must supply a tokenizer when using a non supported language.")

    if self.normalizer_kwargs is None:
      self.normalizer_kwargs = {}

  def calc_global_values(self, array, verbose=True):
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
    self.input_dtype = array.dtype

    if self.index_to_word is None:
      # Get all the words and the number of times they appear
      all_words = {}

      # Define the key word arguments for the normalizer, ensuring that inverse
      # is set to false.
      kwargs = {}
      kwargs.update(self.normalizer_kwargs)
      kwargs['inverse'] = False

      # Tokenize each request, add the tokens to the set of all words
      for string in array[:, self.col_index]:
        tokens = self.normalizer(string)
        for token_num, token in enumerate(tokens):
          all_words.setdefault(token, 0)
          all_words[token] += 1

      # Sort the dict by the number of times the words appear
      sorted_words = sorted(all_words.items(), key=operator.itemgetter(1), reverse=True)

      # Pull out the first 'max_vocab_size' words
      sorted_words = [w for w, c in sorted_words[:self.max_vocab_size - 1]]

      # Create the mapping from category values to index in the vector and
      # vice versa
      self.index_to_word = sorted(sorted_words)
      self.index_to_word = ['__UNK__'] + self.index_to_word

    # Ensure the max_vocab_size agrees with the index_to_word, and define the
    # reverse mapping word_to_index.
    self.max_vocab_size = len(self.index_to_word)
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

  def forward_transform(self, array, verbose=True):
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
    indices = -1 * np.ones([array.shape[0], self.max_sent_len], dtype=np.int64)
    token_num = 0
    string = row[column]
    tokens = self._normalize_string(string)
    for token in tokens:
      # If the max size has been reached then break.
      if token_num == self.mat_sent_len:
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
    return self.max_sent_len
