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


class MultiLingualStringTransform(n.Transform):
  """Object that transforms raw strings into vectorized data with all the necessary data needed in order to completely reconstruct the original strings.

  Parameters
  ----------
  name : str
    The name of the transform.
  from_file : str
    The path to the saved file to recreate the transform object that was saved to disk.
  save_dict : dict
    The dictionary to recreate the transform object
  lower_case : bool
    Whether or not to lower_case all the strings.
  half_width : bool
    Whether or not to convert all full width characters to half width.
  lemmatize : bool
    Whether or not to stem or lemmatize the words.
  index_to_word : list
    The mapping from index number to word.
  word_to_index : dict
    The mapping from word to index number.
  max_sent_len : int
    The maximum allowed number of words in a sentence. Also decides the inner most dimension of the outputted indices array.
  max_vocab_size : int
    The maximum allowed number of words in the vocabulary.
  unk_index : int
    The location of the 'unknown token' in the index_to_word mapping.
  word_tokenizer : func
    A function that takes in a string and splits it up into a list of words.
  word_detokenizer : func
    A function that takes in a list of words and outputs a string. Doesn't have to be an exact inverse to word_tokenizer but should be close otherwise a lot of large diff strings will have to be outputted in order to reproduce the original strings.

  Attributes
  ----------
  input_dtype: numpy dtype
    The datatype of the original inputted array.
  input_shape: list of ints
    The shape of the original inputted array.

  """

  attribute_dict = {'name': '', 'dtype': np.int64, 'input_shape': None, 'index_to_word_maps': None, 'word_to_index_maps': None, 'max_sent_len': None, 'word_tokenizers': None, 'lemmatize': False, 'lemmatizer': None, 'half_width': False, 'lower_case': False, 'word_detokenizers': None, 'max_vocab_size': None}

  for k, v in n.Transform.attribute_dict.iteritems():
    if k in attribute_dict:
      continue
    attribute_dict[k] = v

  required_params = set(['word_tokenizers', 'word_detokenizers'])
  required_params.update(n.Transform.required_params)


  def __init__(self, from_file=None, save_dict=None, **kwargs):
    super(MultiLingualStringTransform, self).__init__(from_file, save_dict, **kwargs)

    if self.index_to_word_maps is None and self.max_vocab_size is None:
      raise ValueError("Must supply index_to_word_maps mapping or a max_vocab_size.")
    if self.word_tokenizers is None:
      raise ValueError("No tokenizers set for this Transform. Must supply one per language.")
    if self.word_detokenizers is None:
      raise ValueError("No detokenizers set for this Transform. Must supply one per language.")
    if self.max_sent_len is None:
      raise ValueError("Must specify a max_sent_len.")

  def __len__(self):
    return self.max_sent_len

  def _get_array_attributes(self, prefix=''):
    """Get the dictionary that contain the original shapes of the arrays before being converted into tfrecord examples.

    Parameters
    ----------
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    dict
      The dictionary with keys equal to those that are found in the Transform's example dicts and values are the shapes of the arrays of a single example.

    """
    # Create the list of all the keys expected from the example_dicts
    array_keys = self._get_array_keys()

    # Get the original array's shape, except for the batch dim.
    shape = [1]

    att_dict = {}
    for key in array_keys:

      # Add a max_sent_len dim to the tokenize_diff array since it has a
      # diff for each token, otherwise give it input array's shape.
      cur_shape = shape if key in ('tokenize_diff', 'languages') else shape + [self.max_sent_len]

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
    array_keys = ['indices', 'tokenize_diff', 'missing_vals', 'languages']
    if self.lower_case:
      array_keys.append('lower_case_diff')
    if self.half_width:
      array_keys.append('half_width_diff')
    if self.lemmatize:
      array_keys.append('lemmatize_diff')
    return array_keys

  def _save_dict(self):
    """Create the dictionary of values needed in order to reconstruct the transform."""
    save_dict = {}
    for key in self.attribute_dict:
      save_dict[key] = getattr(self, key)
    save_dict['__class__'] = str(self.__class__.__name__)
    return save_dict

  def _start_calc(self):
    # Create the mapping from category values to index in the vector and
    # vice versa
    self.all_words = {l: {} for l in self.word_tokenizers}

  def _finish_calc(self):
    if self.index_to_word_maps is None:
      self.index_to_word_maps = {}
      for language in sorted(self.word_tokenizers):
        # Sort the dict by the number of times the words appear
        sorted_words = sorted(self.all_words[language].items(), key=operator.itemgetter(1), reverse=True)

        # Pull out the first 'max_vocab_size' words
        corrected_vocab_size = self.max_vocab_size - 1
        sorted_words = [w for w, c in sorted_words[:corrected_vocab_size]]

        # Create the mapping from category values to index in the vector and
        # vice versa
        self.index_to_word_maps[language] = ['[UNK]'] + sorted(sorted_words)
    else:
      self.max_vocab_size = max([len(self.index_to_word_maps[k]) for k in self.word_tokenizers])
      for language in sorted(self.word_tokenizers):
        token = self.index_to_word_maps[language][0]
        if token != '[UNK]':
          raise ValueError("First element of the every index_to_word_map must be the '[UNK]' token. Got {} for {}".format(token, language))

    self.word_to_index_maps = {}
    for language in sorted(self.word_tokenizers):
      self.word_to_index_maps[language] = {
        word: num for num, word in enumerate(self.index_to_word_maps[language])
      }

  def _calc_global_values(self, array):
    """Calculate all the values of the Transform that are dependent on all the examples of the dataset. (e.g. mean, standard deviation, unique category values, etc.) This method must be run before any actual transformation can be done.

    Parameters
    ----------
    array : np.ndarray
      The entire dataset.

    """
    if self.dtype is None:
      self.dtype = array.dtype
    else:
      array = np.array(array, dtype=self.input_dtype)
    if len(array.shape) < 2 or array.shape[1] != 2:
      raise ValueError("Array must have exactly two columns. The first being the string, and the second being the language.")

    languages = np.unique(array[:, 1])
    for language in languages:
      if language not in self.word_tokenizers or language not in self.word_detokenizers:
        raise ValueError("All languages must appear in both word_tokenizers and word_detokenizers. {} not found".format(language))

    if self.index_to_word_maps is None:
      for language in sorted(self.word_tokenizers):
        mask = array[:, 1: 2] == language
        lang_array = array[:, 0: 1][mask]

        # Get all the words and the number of times they appear
        strings = lang_array.flatten()

        if strings.size == 0:
          continue

        if self.lower_case:
          strings = np.char.lower(strings)

        if self.half_width:
          strings = np.vectorize(_half_width)(strings)

        word_tokenizer = self.word_tokenizers[language]

        for string in strings:
          # Tokenize each request, add the tokens to the set of all words
          tokens = word_tokenizer(string)
          for token_num, token in enumerate(tokens):
            self.all_words[language].setdefault(token, 0)
            self.all_words[language][token] += 1

    else:
      self.max_vocab_size = max([len(v) for v in self.index_to_word_maps.items()])

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
    splits, splits_slots = td.split(array, [1], axis=1)
    splits_slots['a'].unplug()
    splits_slots['a'].set_name('array')

    splits, _ = td.iter_list(splits['target'], 2)

    # Tokenize the full strings into words
    tokens, tokens_slots = td.multi_tokenize(
      strings=splits[0],
      selector=splits[1],
      tokenizers=self.word_tokenizers,
      detokenizers=self.word_detokenizers,
      max_len=self.max_sent_len
    )

    # Set the names of various tubes and slots to make it easier to reference
    # them in further downstream.
    tokens['diff'].set_name('tokenize_diff')
    tokens_slots['max_len'].set_name('max_sent_len')
    tokens_slots['tokenizers'].set_name('tokenizers')
    tokens_slots['detokenizers'].set_name('detokenizers')

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

    # Lemmatize the strings, and set the diff strings of the tank to
    # 'lemmatize_dff' for easier referencing.
    if self.lemmatize:
      tokens, tokens_slots = td.lemmatize(tokens['target'])
      tokens['diff'].set_name('lemmatize_diff')
      tokens_slots['lemmatizer'].set_name('lemmatizer')

    languages, _ = td.clone(splits[1])
    languages['b'].set_name('languages')

    dim_size, _ = td.dim_size(languages['a'], axis=0)
    shape, _ = td.tube_list(dim_size['target'], 1, 1)
    tile, _ = td.reshape(
      languages['a'], shape['target'],
      tube_plugs={
        'old_shape': lambda z: (z[self._pre('languages', prefix)].shape[0], 1)
      }
    )
    tile, _ = td.tile(
      tile['target'], (1, 1, self.max_sent_len),
      tube_plugs={
        'old_shape': lambda z: (z[self._pre('languages', prefix)].shape[0], 1, 1)
      }
    )

    # Find all the strings which are not in the list of known words and
    # replace them with the 'unknown token'.
    maps_with_empty_strings = {k: v + [''] for k, v in self.index_to_word_maps.iteritems()}
    isin, isin_slots = td.multi_isin(tokens['target'], maps_with_empty_strings, tile['target'])

    mask, _ = td.logical_not(isin['target'])
    tokens, _ = td.replace(
      isin['a'], mask['target'], '[UNK]',
      tube_plugs={
        'mask': lambda z: z[self._pre('indices', prefix)] == 0
      }
    )

    # Keep track values that were overwritten with a 'unknown token'
    tokens['replaced_vals'].set_name('missing_vals')
    isin_slots['bs'].set_name('index_to_word_maps')

    # Convert the tokens into indices.
    indices, indices_slots = td.multi_cat_to_index(
      tokens['target'], tile['target'], self.word_to_index_maps,
      tube_plugs={
        'selector': lambda z: np.tile(np.reshape(z[self._pre('languages')], (z[self._pre('languages')].shape[0], 1, 1)), (1, 1, self.max_sent_len)),
        'missing_vals': lambda z: np.full(z[self._pre('indices')].shape, '', dtype=np.unicode),
        'input_dtype': self.input_dtype
      }
    )

    # Set the names of the slots and tubes of this tank for easier referencing
    indices['target'].set_name('indices')
    # indices['selector'].set_name('languages')
    indices_slots['cat_to_index_maps'].set_name('word_to_index_maps')

    if return_tubes is not None:
      ww = indices['target'].waterwork
      r_tubes = []
      for r_tube_key in return_tubes:
        r_tubes.append(ww.maybe_get_tube(r_tube_key))
      return r_tubes

  def get_schema_dict(self, var_lim=None):

    lang_var_lim = 1
    for lang in self.word_tokenizers:
      if len(lang) > lang_var_lim:
        lang_var_lim = len(lang)
    if var_lim is None:
      var_lim = self.max_sent_len
      for lang in self.index_to_word_maps:
        index_to_word = self.index_to_word_maps[lang]
        for word in index_to_word:
          if len(word) * self.max_sent_len > var_lim:
            var_lim = len(word) * self.max_sent_len

    schema_dict = {self.cols[0]: 'VARCHAR({})'.format(var_lim), self.cols[1]: 'VARCHAR({})'.format(lang_var_lim)}
    return schema_dict
