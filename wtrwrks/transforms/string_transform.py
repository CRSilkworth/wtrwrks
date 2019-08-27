"""Definition of the StringTransform."""
import transform as n
import numpy as np
import wtrwrks.tanks.tank_defs as td
import wtrwrks.read_write.tf_features as feat
from wtrwrks.waterworks.empty import empty
import tensorflow as tf
import operator
import unicodedata


def _half_width(string):
  return unicodedata.normalize('NFKC', unicode(string))


class StringTransform(n.Transform):
  """Object that transforms raw strings into vectorized data with all the necessary data needed in order to completely reconstruct the original strings.

  Parameters
  ----------
  name : str
    The name of the transform.
  dtype : numpy dtype
    The data type the transformed data should have. Defaults to np.float64.
  input_dtype: numpy dtype
    The datatype of the original inputted array.
  lower_case : bool
    Whether or not to lower_case all the strings.
  half_width : bool
    Whether or not to convert all full width characters to half width.
  index_to_word : list
    The mapping from index number to word.
  max_sent_len : int
    The maximum allowed number of words in a sentence. Also decides the inner most dimension of the outputted indices array.
  max_vocab_size : int
    The maximum allowed number of words in the vocabulary.
  word_tokenizer : func
    A function that takes in a string and splits it up into a list of words.
  word_detokenizer : func
    A function that takes in a list of words and outputs a string. Doesn't have to be an exact inverse to word_tokenizer but should be close otherwise a lot of large diff strings will have to be outputted in order to reproduce the original strings.

  Attributes
  ----------
  attribute_dict: dict
    The keys are the attributes of the class while the values are the default values. It's done this way rather than defined in the __init__ because this dictionary also defines what values need to be saved when written to disk, and what values need to be displayed when printed to the terminal.
  required_params: set of strs
    The parameters that must be provided to the transform at definition.
  cols : list of strs
    The column names of the array or dataframe to be transformed.
  num_examples : int
    The number of rows that have passed through the calc_global_values function.
  is_calc_run : bool
    Whether or not calc_global_values has been run for this transform.
  word_to_index : dict
    The mapping from word to index number.

  """

  attribute_dict = {'name': '', 'dtype': np.int64, 'input_shape': None, 'index_to_word': None, 'word_to_index': None, 'max_sent_len': None, 'word_tokenizer': lambda a: a.split(), 'half_width': False, 'lower_case': False, 'word_detokenizer': lambda a: ' '.join(a), 'max_vocab_size': None}

  for k, v in n.Transform.attribute_dict.iteritems():
    if k in attribute_dict:
      continue
    attribute_dict[k] = v

  required_params = set(['max_sent_len'])
  required_params.update(n.Transform.required_params)

  def __init__(self, from_file=None, save_dict=None, **kwargs):
    """Define a transform using a dictionary, file, or by setting the attribute values in kwargs.

    Parameters
    ----------
    from_file : None or str
      The file path of the Tranform that was written to disk.
    save_dict : dict or None
      The dictionary of attributes that completely define a Transform.
    name : str
      The name of the transform.
    dtype : numpy dtype
      The data type the transformed data should have. Defaults to np.float64.
    input_dtype: numpy dtype
      The datatype of the original inputted array.
    lower_case : bool
      Whether or not to lower_case all the strings.
    half_width : bool
      Whether or not to convert all full width characters to half width.
    index_to_word : list
      The mapping from index number to word.
    max_sent_len : int
      The maximum allowed number of words in a sentence. Also decides the inner most dimension of the outputted indices array.
    max_vocab_size : int
      The maximum allowed number of words in the vocabulary.
    word_tokenizer : func
      A function that takes in a string and splits it up into a list of words.
    word_detokenizer : func
      A function that takes in a list of words and outputs a string. Doesn't have to be an exact inverse to word_tokenizer but should be close otherwise a lot of large diff strings will have to be outputted in order to reproduce the original strings.
    **kwargs :
      The keyword arguments that set the values of the attributes defined in the attribute_dict.

    """
    super(StringTransform, self).__init__(from_file, save_dict, **kwargs)

    # Require either a index to word mapping or a max vocab size if it is to be
    # built from scratch.
    if self.index_to_word is None and self.max_vocab_size is None:
      raise ValueError("Must supply index_to_word mapping or a max_vocab_size.")

  def __len__(self):
    return self.max_sent_len

  def _calc_global_values(self, array):
    """Calculate all the values of the Transform that are dependent on all the examples of the dataset. (e.g. mean, standard deviation, unique category values, etc.) This method must be run before any actual transformation can be done.

    Parameters
    ----------
    array : np.ndarray
      Some of the data that will be transformed.

    """
    if self.dtype is None:
      self.dtype = array.dtype
    else:
      array = np.array(array, dtype=self.input_dtype)
    if self.index_to_word is None:
      # Get all the words and the number of times they appear
      strings = array.flatten()
      if self.lower_case:
        strings = np.char.lower(strings)

      if self.half_width:
        strings = np.vectorize(_half_width)(strings)

      for string in strings:
        # Tokenize each request, add the tokens to the set of all words
        tokens = self.word_tokenizer(string)
        for token_num, token in enumerate(tokens):
          self.all_words.setdefault(token, 0)
          self.all_words[token] += 1

  def _finish_calc(self):
    """Finish up the calc global value process."""
    if self.index_to_word is None:
      # Sort the dict by the number of times the words appear
      sorted_words = sorted(self.all_words.items(), key=operator.itemgetter(1), reverse=True)

      # Pull out the first 'max_vocab_size' words
      corrected_vocab_size = self.max_vocab_size - 1
      sorted_words = [w for w, c in sorted_words[:corrected_vocab_size]]

      # Create the mapping from category values to index in the vector and
      # vice versa
      self.index_to_word = sorted(sorted_words)
      self.index_to_word = ['[UNK]'] + self.index_to_word
    else:
      self.max_vocab_size = len(self.index_to_word)

    self.word_to_index = {
      word: num for num, word in enumerate(self.index_to_word)
    }

  def _get_array_attributes(self, prefix=''):
    """Get the dictionary that contain the original shapes of the arrays before being converted into tfrecord examples.

    Parameters
    ----------
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    array_attributes : dict
      The dictionary with keys equal to those that are found in the Transform's example dicts and values are the shapes of the arrays of a single example.

    """
    # Create the list of all the keys expected from the example_dicts
    array_keys = self._get_array_keys()

    # Get the original array's shape, except for the batch dim.
    shape = list([len(self.cols)])

    att_dict = {}
    for key in array_keys:

      # Add a max_sent_len dim to the tokenize_diff array since it has a
      # diff for each token, otherwise give it input array's shape.
      cur_shape = shape if key == 'tokenize_diff' else shape + [self.max_sent_len]

      att_dict[key] = {
        'shape': list(cur_shape),
        'tf_type': tf.int64 if key == 'indices' else tf.string,
        'size': feat.size_from_shape(cur_shape),
        'feature_func': feat._int_feat if key == 'indices' else feat._bytes_feat,
        'np_type': np.int64 if key == 'indices' else np.unicode
      }
    att_dict = self._pre(att_dict, prefix)
    return att_dict

  def _get_array_keys(self):
    """Get the relevant list of keys for this instance of the string transform"""
    array_keys = ['indices', 'tokenize_diff', 'missing_vals']
    if self.lower_case:
      array_keys.append('lower_case_diff')
    if self.half_width:
      array_keys.append('half_width_diff')
    return array_keys

  def _save_dict(self):
    """Create the dictionary of values needed in order to reconstruct the transform."""
    save_dict = {}
    for key in self.attribute_dict:
      save_dict[key] = getattr(self, key)
    save_dict['__class__'] = str(self.__class__.__name__)
    return save_dict

  def _start_calc(self):
    """Start the calc global value process."""
    # Create the mapping from category values to index in the vector and
    # vice versa
    self.all_words = {}

  def define_waterwork(self, array=empty, return_tubes=None, prefix=''):
    """Get the waterwork that completely describes the pour and pump transformations.

    Parameters
    ----------
    array : np.ndarray or empty
      The array to be transformed.

    Returns
    -------
    Waterwork
      The waterwork with all the tanks (operations) added, and names set.

    """
    # Tokenize the full strings into words
    tokens, tokens_slots = td.tokenize(strings=array, tokenizer=self.word_tokenizer, detokenizer=self.word_detokenizer, max_len=self.max_sent_len)
    tokens_slots['strings'].unplug()

    # Set the names of various tubes and slots to make it easier to reference
    # them in further downstream.
    tokens['diff'].set_name('tokenize_diff')
    tokens_slots['max_len'].set_name('max_sent_len')
    tokens_slots['strings'].set_name('array')
    tokens_slots['tokenizer'].set_name('tokenizer')
    tokens_slots['detokenizer'].set_name('detokenizer')

    # lower_case the strings, and set the diff strings of the tank to
    # 'lower_case_dff' for easier referencing.
    if self.lower_case:
      tokens, tokens_slots = td.lower_case(tokens['target'])
      tokens['diff'].set_name('lower_case_diff')

    # Half width the strings, and set the diff strings of the tank to
    # 'half_width_diff' for easier referencing.
    if self.half_width:
      tokens, tokens_slots = td.half_width(tokens['target'])
      tokens['diff'].set_name('half_width_diff')

    # Find all the strings which are not in the list of known words and
    # replace them with the 'unknown token'.
    isin, isin_slots = td.isin(tokens['target'], self.index_to_word + [''])
    mask, _ = td.logical_not(isin['target'])
    tokens, _ = td.replace(
      isin['a'], mask['target'], self.index_to_word[0],
      tube_plugs={
        'mask': lambda z: z[self._pre('indices', prefix)] == 0
      }
    )

    # Keep track values that were overwritten with a 'unknown token'
    tokens['replaced_vals'].set_name('missing_vals')
    isin_slots['b'].set_name('index_to_word')

    # Convert the tokens into indices.
    indices, indices_slots = td.cat_to_index(
      tokens['target'], self.word_to_index,
      tube_plugs={
        'missing_vals': lambda z: np.full(z[self._pre('indices', prefix)].shape, '', dtype=np.unicode),
        'input_dtype': self.input_dtype
      }
    )

    # Set the names of the slots and tubes of this tank for easier referencing
    indices['target'].set_name('indices')
    indices_slots['cat_to_index_map'].set_name('word_to_index')

    if return_tubes is not None:
      ww = indices['target'].waterwork
      r_tubes = []
      for r_tube_key in return_tubes:
        r_tubes.append(ww.maybe_get_tube(r_tube_key))
      return r_tubes

  def get_schema_dict(self, var_lim=None):
    """Create a dictionary which defines the proper fields which are needed to store the untransformed data in a (postgres) SQL database.

    Parameters
    ----------
    var_lim : int or dict
      The maximum size of strings. If an int is passed then all VARCHAR fields have the same limit. If a dict is passed then each field gets it's own limit. Defaults to 255 for all string fields.

    Returns
    -------
    schema_dict : dict
      Dictionary where the keys are the field names and the values are the SQL data types.

    """
    if var_lim is None:
      var_lim = self.max_sent_len
      for word in self.index_to_word:
        if len(word) * self.max_sent_len > var_lim:
          var_lim = len(word) * self.max_sent_len

    return super(StringTransform, self).get_schema_dict(var_lim)
