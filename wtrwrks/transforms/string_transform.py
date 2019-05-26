"""Definition of the StringTransform."""
import transform as n
import numpy as np
import wtrwrks.tanks.tank_defs as td
import wtrwrks.read_write.tf_features as feat
from wtrwrks.waterworks.empty import empty
import tensorflow as tf


class StringTransform(n.Transform):
  """Object that transforms raw strings into vectorized data with all the necessary data needed in order to completely reconstruct the original strings.

  Parameters
  ----------
  name : str
    The name of the transform.
  from_file : str
    The path to the saved file to recreate the transform object that was saved to disk.
  save_dict : dict
    The dictionary to recreate the transform object
  lowercase : bool
    Whether or not to lowercase all the strings.
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

  attribute_dict = {'name': '', 'dtype': np.int64, 'input_dtype': None, 'input_shape': None, 'index_to_word': None, 'word_to_index': None, 'max_sent_len': None, 'word_tokenizer': None, 'lemmatize': False, 'lemmatizer': None, 'half_width': False, 'lower_case': False, 'unk_index': None, 'word_detokenizer': lambda a: ' '.join(a)}

  def __len__(self):
    return self.max_sent_len

  def _extract_pour_outputs(self, tap_dict, prefix=''):
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
    r_dict = {k: tap_dict[self._pre(k, prefix)] for k in ['indices', 'missing_vals', 'tokenize_diff']}
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
    array_keys = ['indices', 'tokenize_diff', 'missing_vals']
    if self.lower_case:
      array_keys.append('lower_case_diff')
    if self.half_width:
      array_keys.append('half_width_diff')
    if self.lemmatize:
      array_keys.append('lemmatize_diff')

    size = np.prod(self.input_shape[1:])

    feature_dict = {}
    for key in array_keys:
      if key == 'indices':
        dtype = tf.int64
      else:
        dtype = tf.string

      if key == 'tokenize_diff':
        shape = [size]
      else:
        shape = [size * self.max_sent_len]
      feature_dict[key] = tf.FixedLenFeature(shape, dtype)

    feature_dict = self._pre(feature_dict, prefix)
    return feature_dict

  def _get_example_dicts(self, pour_outputs, prefix=''):
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
    mask = pour_outputs['indices'] == self.unk_index
    missing_vals = pour_outputs['missing_vals']

    # Convert the 1D missing vals array into a full array of the same size as
    # the indices array. This is so it can be easily separated into individual
    # rows that be put into separate examples.
    pour_outputs['missing_vals'] = self._full_missing_vals(mask, missing_vals)

    array_keys = ['indices', 'tokenize_diff', 'missing_vals']
    if self.lower_case:
      array_keys.append('lower_case_diff')
    if self.half_width:
      array_keys.append('half_width_diff')
    if self.lemmatize:
      array_keys.append('lemmatize_diff')

    # Create an example dict for each row of indices.
    num_examples = pour_outputs['indices'].shape[0]
    example_dicts = []
    for row_num in xrange(num_examples):
      example_dict = {}
      for key in array_keys:

        # Flatten the arrays since tfrecords can only have one dimension.
        # Convert to the appropriate feature type.
        serial = pour_outputs[key][row_num].flatten()
        if key == 'indices':
          example_dict[key] = feat._int_feat(serial)
        else:
          example_dict[key] = feat._bytes_feat(serial)

      example_dict = self._pre(example_dict, prefix)
      example_dicts.append(example_dict)
    return example_dicts

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
      funnel_dict = {'input': array, 'word_to_index': self.word_to_index}
    else:
      funnel_dict = {}

    funnel_dict['index_to_word'] = self.index_to_word + ['']
    funnel_dict['tokenizer'] = self.word_tokenizer
    funnel_dict['detokenizer'] = self.word_detokenizer

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
    pour_outputs = self._nopre(pour_outputs, prefix)

    # Set all the tap values that are common across all string transforms.
    tap_dict = {
      'indices': pour_outputs['indices'],
      'missing_vals': pour_outputs['missing_vals'],
      ('CatToIndex_0', 'cat_to_index_map'): self.word_to_index,
      ('CatToIndex_0', 'input_dtype'): self.input_dtype,
      ('Tokenize_0', 'detokenizer'): self.word_detokenizer,
      ('Tokenize_0', 'tokenizer'): self.word_tokenizer,
      ('Tokenize_0', 'diff'): pour_outputs['tokenize_diff']
    }

    # Set the taps associated with the optional additional operation of the
    # Transform.
    if self.lower_case:
      u_dict = {('LowerCase_0', 'diff'): pour_outputs['lower_case_diff']}
      tap_dict.update(u_dict)
    if self.half_width:
      u_dict = {('HalfWidth_0', 'diff'): pour_outputs['half_width_diff']}
      tap_dict.update(u_dict)
    if self.lemmatize:
      u_dict = {
        ('Lemmatize_0', 'diff'): pour_outputs['lemmatize_diff'],
        ('Lemmatize_0', 'lemmatizer'): self.lemmatizer
      }
      tap_dict.update(u_dict)

    # Find all the empty strings and locations of the unknown values.
    empties = pour_outputs['indices'][pour_outputs['indices'] == -1]
    mask = pour_outputs['indices'] == self.unk_index

    # Add in the information needed to get back the missing_vals
    u_dict = {
      ('CatToIndex_0', 'missing_vals'): [''] * empties.size,
      ('Replace_0', 'mask'): mask,
      ('Replace_0', 'replace_with_shape'): (1,),
      ('IsIn_0', 'b'): self.index_to_word + ['']
    }
    tap_dict.update(u_dict)

    return self._pre(tap_dict, prefix)

  def _parse_example_dicts(self, example_dicts, prefix=''):
    """Convert the list of example_dicts into the original outputs that came from the pour method.

    Parameters
    ----------
    example_dicts: list of dicts of arrays
      The example dictionaries which the arrays associated with a single example.
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    dict
      The dictionary of all the values outputted by the pour method.

    """
    # Create the list of all the keys expected from the example_dicts
    array_keys = ['indices', 'tokenize_diff', 'missing_vals']
    if self.lower_case:
      array_keys.append('lower_case_diff')
    if self.half_width:
      array_keys.append('half_width_diff')
    if self.lemmatize:
      array_keys.append('lemmatize_diff')

    # Go through each example dict, pull out the arrays associated with each
    # key of pour outputs, reshape them to their proper shape and add them
    # to individual lists so that they can be stacked together.
    pour_outputs = {k: list() for k in array_keys}
    for example_dict in example_dicts:
      for key in array_keys:
        fixed_array = example_dict[self._pre(key, prefix)]

        if key != 'tokenize_diff':
          fixed_array = fixed_array.reshape(list(self.input_shape[1:]) + [self.max_sent_len])
        else:
          fixed_array = fixed_array.reshape(self.input_shape[1:])

        pour_outputs[key].append(fixed_array)

    # Stack the lists of arrays into arrays with a batch dimension
    for key in array_keys:
      pour_outputs[key] = np.stack(pour_outputs[key])

    # Reconstruct the missing values by flattening the stacked array and
    # pulling out only those values which correspond to an unknown value.
    # This gives you the original 1D array of dense missing values for the
    # original array.
    mask = pour_outputs['indices'] == self.unk_index
    pour_outputs['missing_vals'] = pour_outputs['missing_vals'][mask].flatten()

    pour_outputs = self._pre(pour_outputs, prefix)
    return pour_outputs

  def _setattributes(self, **kwargs):
    """Set the actual attributes of the Transform and do some value checks to make sure they valid inputs.

    Parameters
    ----------
    **kwargs :
      The keyword arguments that set the values of the attributes defined in the attribute_dict.

    """
    super(StringTransform, self)._setattributes(**kwargs)

    if self.index_to_word is None:
      raise ValueError("Must supply index_to_word mapping.")
    if self.unk_index is None:
      raise ValueError("Must specify an unk_index. The index to assign the unknown words.")
    if self.word_tokenizer is None:
      raise ValueError("No tokenizer set for this Transform. Must supply one as input into pour.")

  def _shape_def(self, prefix=''):
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
    array_keys = ['indices', 'tokenize_diff', 'missing_vals']
    if self.lower_case:
      array_keys.append('lower_case_diff')
    if self.half_width:
      array_keys.append('half_width_diff')
    if self.lemmatize:
      array_keys.append('lemmatize_diff')

    # Get the original array's shape, except for the batch dim.
    shape = list(self.input_shape[1:])

    shape_dict = {}
    for key in array_keys:
      # Add a max_sent_len dim to the tokenize_diff array since it has a
      # diff for each token, otherwise give it input array's shape.
      shape_dict[key] = shape if key == 'tokenize_diff' else shape + [self.max_sent_len]

    shape_dict = self._pre(shape_dict, prefix)
    return shape_dict

  def calc_global_values(self, array, verbose=True):
    """Calculate all the values of the Transform that are dependent on all the examples of the dataset. (e.g. mean, standard deviation, unique category values, etc.) This method must be run before any actual transformation can be done.

    Parameters
    ----------
    array : np.ndarray
      The entire dataset.
    verbose : bool
      Whether or not to print out warnings.

    """
    self.input_dtype = array.dtype
    self.input_shape = array.shape
    self.word_to_index = {
      word: num for num, word in enumerate(self.index_to_word)
    }
    if self.max_sent_len is None:
      max_len = None
      for string in array.flatten():
        cur_len = len(self.word_tokenizer(string))
        if max_len is None or cur_len > max_len:
          max_len = cur_len
      self.max_sent_len = max_len

  def define_waterwork(self, array=empty):
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

    # Set the names of various tubes and slots to make it easier to reference
    # them in further downstream.
    tokens['diff'].set_name('tokenize_diff')
    tokens_slots['max_len'].set_name('max_sent_len')
    tokens_slots['strings'].set_name('input')
    tokens_slots['tokenizer'].set_name('tokenizer')
    tokens_slots['detokenizer'].set_name('detokenizer')

    # Lowercase the strings, and set the diff strings of the tank to
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

    # Find all the strings which are not in the list of known words and
    # replace them with the 'unknown token'.
    isin, isin_slots = td.isin(tokens['target'], self.index_to_word + [''])
    mask, _ = td.logical_not(isin['target'])
    tokens, _ = td.replace(isin['a'], mask['target'], self.index_to_word[self.unk_index])

    # Keep track values that were overwritten with a 'unknown token'
    tokens['replaced_vals'].set_name('missing_vals')
    isin_slots['b'].set_name('index_to_word')

    # Convert the tokens into indices.
    indices, indices_slots = td.cat_to_index(tokens['target'], self.word_to_index)

    # Set the names of the slots and tubes of this tank for easier referencing
    indices['target'].set_name('indices')
    indices_slots['cat_to_index_map'].set_name('word_to_index')
