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
        r_dict = self.normalizer(string, self.max_sent_len, **kwargs)
        tokens = r_dict['tokens']
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
    # Define the key word arguments for the normalizer, ensuring that inverse
    # is set to false.
    kwargs = {}
    kwargs.update(self.normalizer_kwargs)
    kwargs['inverse'] = False

    # Find the indices for each word, filling with -1 if the vector is
    # longer than the number of tokens.
    indices = -1 * np.ones([array.shape[0], self.max_sent_len], dtype=np.int64)
    diff_string = np.empty([array.shape[0]], dtype=object)
    unks = np.empty([array.shape[0], self.max_sent_len], dtype=object)
    unks.fill('')
    for row_num, string in enumerate(array[:, self.col_index]):
      r_dict = self.normalizer(string, self.max_sent_len, **kwargs)
      tokens = r_dict['tokens']

      for token_num, token in enumerate(tokens):
        # If the word isn't known, fill it with the 'UNK' index.
        # Otherwise pull out the relevant index.
        if token not in self.word_to_index:
          index = self.word_to_index['__UNK__']
          unks[row_num, token_num] = token
        else:
          index = self.word_to_index[token]

        # Fill the array and increase the token number.
        indices[row_num, token_num] = index
      diff_string[row_num] = r_dict['diff_string']
    return {'data': indices, 'diff_string': diff_string.astype(np.unicode), 'unknowns': unks}

  def backward_transform(self, arrays_dict, verbose=True):
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
    kwargs = {}
    kwargs.update(self.normalizer_kwargs)
    kwargs['inverse'] = True

    array = np.empty([arrays_dict['data'].shape[0], 1], dtype=object)
    for row_num, (indices, diff_string, unks) in enumerate(zip(arrays_dict['data'], arrays_dict['diff_string'], arrays_dict['unknowns'])):

      tokens = [self.index_to_word[i] if i != 0 else unks[num] for num, i in enumerate(indices) if i != -1]

      string = self.normalizer({'tokens': tokens, 'diff_string': diff_string}, self.max_sent_len, **kwargs)

      array[row_num] = string

    return array.astype(self.input_dtype)

  def __len__(self):
    """Get the length of the vector outputted by the row_to_vector method."""
    return self.max_sent_len
