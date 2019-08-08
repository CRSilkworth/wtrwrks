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

  attribute_dict = {'name': '', 'dtype': np.int64, 'input_dtype': None, 'input_shape': None, 'index_to_word_maps': None, 'word_to_index_maps': None, 'max_sent_len': None, 'word_tokenizers': None, 'lemmatize': False, 'lemmatizer': None, 'half_width': False, 'lower_case': False, 'unk_index': None, 'word_detokenizers': None, 'max_vocab_size': None}

  attribute_dict.update({k: v for k, v in n.Transform.attribute_dict.iteritem() if k not in attribute_dict})

  required_params = set(['word_tokenizers', 'word_detokenizers'])
  required_params.update(n.Transform.required_params)

  def __len__(self):
    return self.max_sent_len

  def _extract_pour_outputs(self, tap_dict, prefix='', **kwargs):
    """Pull out all the values from tap_dict that cannot be explicitly reconstructed from the transform itself. These are the values that will need to be fed back to the transform into run the tranform in the pump direction.

    Parameters
    ----------
    tap_dict : dict
      The dictionary outputted by the pour (forward) transform.
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    dict
      Dictionay of only those tap dict values which are can't be inferred from the Transform itself.

    """
    r_dict = {k: tap_dict[self._pre(k, prefix)] for k in ['indices', 'missing_vals', 'tokenize_diff', 'languages']}
    if self.lower_case:
      r_dict['lower_case_diff'] = tap_dict[self._pre('lower_case_diff', prefix)]
    if self.half_width:
      r_dict['half_width_diff'] = tap_dict[self._pre('half_width_diff', prefix)]
    if self.lemmatize:
      r_dict['lemmatize_diff'] = tap_dict[self._pre('lemmatize_diff', prefix)]
    r_dict = self._pre(r_dict, prefix)
    return r_dict

  def _extract_pump_outputs(self, funnel_dict, prefix=''):
    """Pull out the original array from the funnel_dict which was produced by running pump.

    Parameters
    ----------
    funnel_dict : dict
      The dictionary outputted by running the transform's pump method. The keys are the names of the funnels and the values are the values of the tubes.
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    np.ndarray
      The array reconstructed from the pump method.

    """
    return funnel_dict[self._pre('input', prefix)]

  def _feature_def(self, num_cols=1, prefix=''):
    """Get the dictionary that contain the FixedLenFeature information for each key found in the example_dicts. Needed in order to build ML input pipelines that read tfrecords.

    Parameters
    ----------
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    dict
      The dictionary with keys equal to those that are found in the Transform's example dicts and values equal the FixedLenFeature defintion of the example key.

    """
    array_keys = self._get_array_keys()

    size = np.prod(self.input_shape[1:])

    feature_dict = {}
    for key in array_keys:
      if key == 'indices':
        dtype = tf.int64
      else:
        dtype = tf.string

      if key == 'tokenize_diff':
        shape = [size]

      elif key == 'languages':
        shape = [1]
      else:
        shape = [size * self.max_sent_len]

      feature_dict[key] = tf.FixedLenFeature(shape, dtype)

    feature_dict = self._pre(feature_dict, prefix)
    return feature_dict

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

  def _alter_pour_outputs(self, pour_outputs, prefix=''):
    """Create a list of dictionaries for each example from the outputs of the pour method.

    Parameters
    ----------
    pour_outputs : dict
      The outputs of the _extract_pour_outputs method.
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    list of dicts of features
      The example dictionaries which contain tf.train.Features.

    """
    pour_outputs = self._nopre(pour_outputs, prefix)

    # Find the locations of all the missing values. i.e. those that have been
    # replace by the unknown token.
    # mask = pour_outputs['indices'] == self.unk_index
    # missing_vals = pour_outputs['missing_vals']

    # Convert the 1D missing vals array into a full array of the same size as
    # the indices array. This is so it can be easily separated into individual
    # rows that be put into separate examples.
    # pour_outputs['missing_vals'] = self._full_missing_vals(mask, missing_vals)
    pour_outputs = self._pre(pour_outputs, prefix)
    return pour_outputs

  def _get_funnel_dict(self, array=None, prefix=''):
    """Construct a dictionary where the keys are the names of the slots, and the values are either values from the Transform itself, or are taken from the supplied array.

    Parameters
    ----------
    array : np.ndarray
      The inputted array of raw information that is to be fed through the pour method.
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    dict
      The dictionary with all funnels filled with values necessary in order to run the pour method.

    """
    if array is not None:
      funnel_dict = {'input': array}
    else:
      funnel_dict = {}

    if self.lemmatize and self.lemmatizer is None:
      raise ValueError("No lemmatizer set for this Transform. Must supply one as input into pour.")
    elif self.lemmatize:
      funnel_dict['lemmatizer'] = self.lemmatizer

    return self._pre(funnel_dict, prefix)

  def _get_tap_dict(self, pour_outputs, prefix=''):
    """Construct a dictionary where the keys are the names of the tubes, and the values are either values from the Transform itself, or are taken from the supplied pour_outputs dictionary.

    Parameters
    ----------
    pour_outputs : dict
      The dictionary of all the values outputted by the pour method.
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    The dictionary with all taps filled with values necessary in order to run the pump method.

    """
    pour_outputs = super(MultiLingualStringTransform, self)._get_tap_dict(pour_outputs, prefix)
    pour_outputs = self._nopre(pour_outputs, prefix)
    # Set all the tap values that are common across all string transforms.
    tap_dict = {
      'indices': pour_outputs['indices'],
      'missing_vals': pour_outputs['missing_vals'],
      'languages': pour_outputs['languages'],
      ('MultiCatToIndex_0', 'input_dtype'): self.input_dtype,
      ('MultiTokenize_0', 'diff'): pour_outputs['tokenize_diff']
    }

    # Set the taps associated with the optional additional operation of the
    # Transform.
    if self.lower_case:
      u_dict = {('lower_case_0', 'diff'): pour_outputs['lower_case_diff']}
      tap_dict.update(u_dict)
    if self.half_width:
      u_dict = {('HalfWidth_0', 'diff'): pour_outputs['half_width_diff']}
      tap_dict.update(u_dict)
    if self.lemmatize:
      u_dict = {
        ('Lemmatize_0', 'diff'): pour_outputs['lemmatize_diff'],
      }
      tap_dict.update(u_dict)

    # Find all the empty strings and locations of the unknown values.
    mask = pour_outputs['indices'] == self.unk_index

    # Add in the information needed to get back the missing_vals
    u_dict = {
      ('MultiCatToIndex_0', 'missing_vals'): np.full(pour_outputs['indices'].shape, '', dtype=np.unicode),
      ('Replace_0', 'mask'): mask,
      # ('Replace_0', 'replace_with_shape'): (1,),
    }
    tap_dict.update(u_dict)

    return self._pre(tap_dict, prefix)

  def _save_dict(self):
    """Create the dictionary of values needed in order to reconstruct the transform."""
    save_dict = {}
    for key in self.attribute_dict:
      save_dict[key] = getattr(self, key)
    save_dict['__class__'] = str(self.__class__.__name__)
    return save_dict

  def _setattributes(self, **kwargs):
    """Set the actual attributes of the Transform and do some value checks to make sure they valid inputs.

    Parameters
    ----------
    **kwargs :
      The keyword arguments that set the values of the attributes defined in the attribute_dict.

    """
    super(MultiLingualStringTransform, self)._setattributes(**kwargs)

    if self.index_to_word_maps is None and self.max_vocab_size is None:
      raise ValueError("Must supply index_to_word_maps mapping or a max_vocab_size.")
    if self.unk_index is None and self.index_to_word_maps is not None:
      raise ValueError("Must specify an unk_index. The index to assign the unknown words.")
    if self.word_tokenizers is None:
      raise ValueError("No tokenizers set for this Transform. Must supply one per language.")
    if self.word_detokenizers is None:
      raise ValueError("No detokenizers set for this Transform. Must supply one per language.")
    if self.max_sent_len is None:
      raise ValueError("Must specify a max_sent_len.")

  def _start_calc(self):
    # Create the mapping from category values to index in the vector and
    # vice versa
    self.all_words = {l: {} for l in self.word_tokenizers}

  def _finish_calc(self):
    if self.index_to_word_maps is None:
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
      self.max_vocab_size = len(self.index_to_word)
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
    if len(array.shape) < 2 or array.shape[1] != 2:
      raise ValueError("Array must have exactly two columns. The first being the string, and the second being the language.")

    languages = np.unique(array[:, 1])
    for language in languages:
      if language not in self.word_tokenizers or language not in self.word_detokenizers:
        raise ValueError("All languages must appear in both word_tokenizers and word_detokenizers. {} not found".format(language))

    if self.index_to_word_maps is None:
      self.index_to_word_maps = {}
      for language in sorted(self.word_tokenizers):
        mask = array[:, 1: 2] == language
        lang_array = array[:, 0: 1][mask]

        if not lang_array.size:
          self.index_to_word_maps[language] = []
          continue

        # Get all the words and the number of times they appear
        strings = lang_array.flatten()
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



  def define_waterwork(self, array=empty, return_tubes=None):
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
    splits_slots['a'].set_name('input')

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
        'old_shape': lambda z: (z[self._pre('languages')].shape[0], 1)
      }
    )
    tile, _ = td.tile(
      tile['target'], (1, 1, self.max_sent_len),
      tube_plugs={
        'old_shape': lambda z: (z[self._pre('languages')].shape[0], 1, 1)
      }
    )

    # Find all the strings which are not in the list of known words and
    # replace them with the 'unknown token'.
    maps_with_empty_strings = {k: v + [''] for k, v in self.index_to_word_maps.iteritems()}
    isin, isin_slots = td.multi_isin(tokens['target'], maps_with_empty_strings, tile['target'])

    mask, _ = td.logical_not(isin['target'])
    tokens, _ = td.replace(isin['a'], mask['target'], '__UNK__')

    # Keep track values that were overwritten with a 'unknown token'
    tokens['replaced_vals'].set_name('missing_vals')
    isin_slots['bs'].set_name('index_to_word_maps')

    # Convert the tokens into indices.
    indices, indices_slots = td.multi_cat_to_index(
      tokens['target'], tile['target'], self.word_to_index_maps,
      tube_plugs={
        'selector': lambda z: np.tile(np.reshape(z[self._pre('languages')], (z[self._pre('languages')].shape[0], 1, 1)), (1, 1, self.max_sent_len)),
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
