import transform as n
import numpy as np
import reversible_transforms.tanks.tank_defs as td
import reversible_transforms.read_write.tf_features as feat
from reversible_transforms.waterworks.empty import empty
import os
import tensorflow as tf
import pprint

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
  attribute_dict = {'name': '', 'dtype': np.int64, 'input_dtype': None, 'input_shape': None, 'index_to_word': None, 'word_to_index': None, 'max_sent_len': None, 'word_tokenizer': None, 'lemmatize': False, 'lemmatizer': None, 'half_width': False, 'lower_case': False, 'unk_index': None, 'word_detokenizer': lambda a: ' '.join(a)}

  def _setattributes(self, **kwargs):
    super(StringTransform, self)._setattributes(**kwargs)

    if self.index_to_word is None:
      raise ValueError("Must supply index_to_word mapping.")
    if self.unk_index is None:
      raise ValueError("Must specify an unk_index. The index to assign the unknown words.")
    if self.word_tokenizer is None:
      raise ValueError("No tokenizer set for this Transform. Must supply one as input into pour.")

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
    tokens, tokens_slots = td.tokenize(strings=array, tokenizer=self.word_tokenizer, detokenizer=self.word_detokenizer, max_len=self.max_sent_len)
    tokens['diff'].set_name('tokenize_diff')
    tokens_slots['max_len'].set_name('max_sent_len')
    tokens_slots['strings'].set_name('input')
    tokens_slots['tokenizer'].set_name('tokenizer')
    tokens_slots['detokenizer'].set_name('detokenizer')

    if self.lower_case:
      tokens, tokens_slots = td.lower_case(tokens['target'])
      tokens['diff'].set_name('lower_case_diff')

    if self.half_width:
      tokens, tokens_slots = td.half_width(tokens['target'])
      tokens['diff'].set_name('half_width_diff')

    if self.lemmatize:
      tokens, tokens_slots = td.lemmatize(tokens['target'])
      tokens['diff'].set_name('lemmatize_diff')
      tokens_slots['lemmatizer'].set_name('lemmatizer')

    isin, isin_slots = td.isin(tokens['target'], self.index_to_word + [''])
    mask, _ = td.logical_not(isin['target'])
    tokens, _ = td.replace(isin['a'], mask['target'], self.index_to_word[self.unk_index])
    tokens['replaced_vals'].set_name('missing_vals')
    isin_slots['b'].set_name('index_to_word')

    indices, indices_slots = td.cat_to_index(tokens['target'], self.word_to_index)
    indices['target'].set_name('indices')
    indices_slots['cat_to_index_map'].set_name('word_to_index')
    if self.unk_index is None:
      indices['missing_vals'].set_name('missing_vals')

  def _get_funnel_dict(self, array=None, prefix=''):
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

  def _extract_pour_outputs(self, tap_dict, prefix=''):
    r_dict = {k: tap_dict[self._pre(k, prefix)] for k in ['indices', 'missing_vals', 'tokenize_diff']}
    if self.lower_case:
      r_dict['lower_case_diff'] = tap_dict[self._pre('lower_case_diff', prefix)]
    if self.half_width:
      r_dict['half_width_diff'] = tap_dict[self._pre('half_width_diff', prefix)]
    if self.lemmatize:
      r_dict['lemmatize_diff'] = tap_dict[self._pre('lemmatize_diff', prefix)]
    r_dict = self._pre(r_dict, prefix)
    return r_dict

  def _get_tap_dict(self, pour_outputs, prefix=''):
    pour_outputs = self._nopre(pour_outputs, prefix)
    tap_dict = {
      'indices': pour_outputs['indices'],
      'missing_vals': pour_outputs['missing_vals'],
      ('CatToIndex_0', 'cat_to_index_map'): self.word_to_index,
      ('CatToIndex_0', 'input_dtype'): self.input_dtype,
      ('Tokenize_0', 'detokenizer'): self.word_detokenizer,
      ('Tokenize_0', 'tokenizer'): self.word_tokenizer,
      ('Tokenize_0', 'diff'): pour_outputs['tokenize_diff']
    }
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
    if self.unk_index is not None:
      empties = pour_outputs['indices'][pour_outputs['indices'] == -1]
      mask = pour_outputs['indices'] == self.unk_index
      u_dict = {
        ('CatToIndex_0', 'missing_vals'): [''] * empties.size,
        ('Replace_0', 'mask'): mask,
        ('Replace_0', 'replace_with_shape'): (1,),
        ('IsIn_0', 'b'): self.index_to_word + ['']
      }
      tap_dict.update(u_dict)

    return self._pre(tap_dict, prefix)

  def _extract_pump_outputs(self, funnel_dict, prefix=''):
    return funnel_dict[self._pre('input', prefix)]

  def _get_example_dicts(self, pour_outputs, prefix=''):
    pour_outputs = self._nopre(pour_outputs, prefix)
    mask = pour_outputs['indices'] == self.unk_index
    missing_vals = pour_outputs['missing_vals']

    pour_outputs['missing_vals'] = self._full_missing_vals(mask, missing_vals)

    array_keys = ['indices', 'tokenize_diff', 'missing_vals']
    if self.lower_case:
      array_keys.append('lower_case_diff')
    if self.half_width:
      array_keys.append('half_width_diff')
    if self.lemmatize:
      array_keys.append('lemmatize_diff')

    num_examples = pour_outputs['indices'].shape[0]
    example_dicts = []
    for row_num in xrange(num_examples):
      example_dict = {}
      for key in array_keys:
        serial = pour_outputs[key][row_num].flatten()
        if key == 'indices':
          example_dict[key] = feat._int_feat(serial)
        else:
          example_dict[key] = feat._bytes_feat(serial)

      example_dict = self._pre(example_dict, prefix)
      example_dicts.append(example_dict)
    return example_dicts

  def _parse_example_dicts(self, example_dicts, prefix=''):
    array_keys = ['indices', 'tokenize_diff', 'missing_vals']
    if self.lower_case:
      array_keys.append('lower_case_diff')
    if self.half_width:
      array_keys.append('half_width_diff')
    if self.lemmatize:
      array_keys.append('lemmatize_diff')

    pour_outputs = {k: list() for k in array_keys}
    for example_dict in example_dicts:
      for key in array_keys:
        fixed_array = example_dict[self._pre(key, prefix)]

        if key != 'tokenize_diff':
          fixed_array = fixed_array.reshape(list(self.input_shape[1:]) + [self.max_sent_len])
        else:
          fixed_array = fixed_array.reshape(self.input_shape[1:])

        pour_outputs[key].append(fixed_array)

    for key in array_keys:
      pour_outputs[key] = np.stack(pour_outputs[key])

    mask = pour_outputs['indices'] == self.unk_index
    pour_outputs['missing_vals'] = pour_outputs['missing_vals'][mask].flatten()

    pour_outputs = self._pre(pour_outputs, prefix)
    return pour_outputs

  def _feature_def(self, num_cols=1, prefix=''):
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

  def _shape_def(self, prefix=''):
    array_keys = ['indices', 'tokenize_diff', 'missing_vals']
    if self.lower_case:
      array_keys.append('lower_case_diff')
    if self.half_width:
      array_keys.append('half_width_diff')
    if self.lemmatize:
      array_keys.append('lemmatize_diff')

    shape = list(self.input_shape[1:])

    shape_dict = {}
    for key in array_keys:
      shape_dict[key] = shape if key == 'tokenize_diff' else shape + [self.max_sent_len]

    shape_dict = self._pre(shape_dict, prefix)
    return shape_dict

  def __len__(self):
    """Get the length of the vector outputted by the row_to_vector method."""
    return self.max_sent_len
