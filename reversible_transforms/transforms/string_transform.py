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
  attribute_dict = {'name': '', 'dtype': np.int64, 'input_dtype': None, 'index_to_word': None, 'word_to_index': None, 'max_sent_len': 20, 'tokenizer': None, 'lemmatize': False, 'lemmatizer': None, 'half_width': False, 'lower_case': False, 'unk_index': None, 'delimiter': ' '}

  def _setattributes(self, **kwargs):
    super(StringTransform, self)._setattributes(**kwargs)

    if self.index_to_word is None:
      raise ValueError("Must supply index_to_word mapping.")
    # if self.index_to_word is not None and self.index_to_word[0] != '__UNK__':
    #   raise ValueError("First element of the index_to_word map must be the default '__UNK__' token")

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
    self.word_to_index = {
      word: num for num, word in enumerate(self.index_to_word)
    }

  def define_waterwork(self, array=empty):
    tokens, tokens_slots = td.tokenize(strings=array, max_len=self.max_sent_len)
    tokens['diff'].set_name('tokenize_diff')
    tokens_slots['max_len'].set_name('max_sent_len')
    tokens_slots['strings'].set_name('input')
    tokens_slots['tokenizer'].set_name('tokenizer')
    tokens_slots['delimiter'].set_name('delimiter')

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

    if self.unk_index is not None:
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

  def _get_funnel_dict(self, array=None, tokenizer=None, delimiter=None, lemmatizer=None, prefix=''):
    if array is not None:
      funnel_dict = {'input': array, 'word_to_index': self.word_to_index}
    else:
      funnel_dict = {}

    if self.unk_index is not None:
      funnel_dict['index_to_word'] = self.index_to_word + ['']

    if tokenizer is None and self.tokenizer is None:
      raise ValueError("No tokenizer set for this Transform. Must supply one as input into pour.")
    elif tokenizer is not None:
      funnel_dict['tokenizer'] = tokenizer
    else:
      funnel_dict['tokenizer'] = self.tokenizer

    if delimiter is not None:
      funnel_dict['delimiter'] = delimiter
    else:
      funnel_dict['delimiter'] = self.delimiter

    if self.lemmatize and lemmatizer is None and self.lemmatizer is None:
      raise ValueError("No lemmatizer set for this Transform. Must supply one as input into pour.")
    elif self.lemmatize and lemmatizer is not None:
      funnel_dict['lemmatizer'] = lemmatizer
    elif self.lemmatize:
      funnel_dict['lemmatizer'] = self.lemmatizer

    return self._add_name_to_dict(funnel_dict, prefix)

  def _extract_pour_outputs(self, tap_dict, prefix=''):
    r_dict = {k: tap_dict[self._add_name(k, prefix)] for k in ['indices', 'missing_vals', 'tokenize_diff']}
    if self.lower_case:
      r_dict['lower_case_diff'] = tap_dict[self._add_name('lower_case_diff', prefix)]
    if self.half_width:
      r_dict['half_width_diff'] = tap_dict[self._add_name('half_width_diff', prefix)]
    if self.lemmatize:
      r_dict['lemmatize_diff'] = tap_dict[self._add_name('lemmatize_diff', prefix)]

    return r_dict

  def _get_tap_dict(self, indices, missing_vals, tokenize_diff, lower_case_diff=None, half_width_diff=None, lemmatize_diff=None, delimiter=None, prefix=''):
    if delimiter is None:
      delimiter = self.delimiter

    tap_dict = {
      'indices': indices,
      'missing_vals': missing_vals,
      ('CatToIndex_0', 'cat_to_index_map'): self.word_to_index,
      ('CatToIndex_0', 'input_dtype'): self.input_dtype,
      ('Tokenize_0', 'delimiter'): delimiter,
      ('Tokenize_0', 'tokenizer'): self.tokenizer,
      ('Tokenize_0', 'diff'): tokenize_diff
    }
    if self.lower_case:
      u_dict = {('LowerCase_0', 'diff'): lower_case_diff}
      tap_dict.update(u_dict)
    if self.half_width:
      u_dict = {('HalfWidth_0', 'diff'): half_width_diff}
      tap_dict.update(u_dict)
    if self.lemmatize:
      u_dict = {
        ('Lemmatize_0', 'diff'): lemmatize_diff,
        ('Lemmatize_0', 'lemmatizer'): self.lemmatizer
      }
      tap_dict.update(u_dict)
    if self.unk_index is not None:
      u_dict = {
        ('CatToIndex_0', 'missing_vals'): [''] * indices[indices == -1].size,
        ('Replace_0', 'mask'): indices == self.unk_index,
        ('Replace_0', 'replace_with_shape'): (1,),
        ('IsIn_0', 'b'): self.index_to_word + ['']
      }
      tap_dict.update(u_dict)

    return self._add_name_to_dict(tap_dict, prefix)

  def _extract_pump_outputs(self, funnel_dict, prefix=''):
    return funnel_dict[self._add_name('input', prefix)]

  def pour_examples(self, array, tokenizer=None, delimiter=None, lemmatizer=None):
    ww = self.get_waterwork()
    funnel_dict = self._get_funnel_dict(array, tokenizer, delimiter, lemmatizer)
    tap_dict = ww.pour(funnel_dict, key_type='str')

    pour_outputs = self._extract_pour_outputs(tap_dict)

    mask = pour_outputs['indices'] == self.unk_index
    missing_vals = pour_outputs['missing_vals'].tolist()
    pour_outputs['missing_vals'] = self._full_missing_vals(mask, missing_vals)

    array_keys = ['indices', 'tokenize_diff', 'missing_vals']
    if self.lower_case:
      array_keys.append('lower_case_diff')
    if self.half_width:
      array_keys.append('half_width_diff')
    if self.lemmatize:
      array_keys.append('lemmatize_diff')

    example_dicts = []
    for row_num in xrange(array.shape[0]):
      example_dict = {}
      for key in array_keys:
        serial = pour_outputs[key][row_num].flatten()
        if key == 'indices':
          example_dict[key] = feat._int_feat(serial)
        else:
          example_dict[key] = feat._bytes_feat(serial)

      example_dicts.append(example_dict)

    return example_dicts

  def pump_examples(self, example_dicts, prefix=''):
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
        fixed_array = example_dict[key]

        if key != 'tokenize_diff':
          fixed_array = fixed_array.reshape([-1, self.max_sent_len])

        pour_outputs[key].append(fixed_array)

    for key in array_keys:
      pour_outputs[key] = np.stack(pour_outputs[key])

    mask = pour_outputs['indices'] == self.unk_index
    pour_outputs['missing_vals'] = pour_outputs['missing_vals'][mask].flatten()

    ww = self.get_waterwork()
    tap_dict = self._get_tap_dict(**pour_outputs)
    funnel_dict = ww.pump(tap_dict, key_type='str')
    return self._extract_pump_outputs(funnel_dict)

  def _feature_def(self, num_cols=1):
    # Create the dictionary defining the structure of the example
    array_keys = ['indices', 'tokenize_diff', 'missing_vals']
    if self.lower_case:
      array_keys.append('lower_case_diff')
    if self.half_width:
      array_keys.append('half_width_diff')
    if self.lemmatize:
      array_keys.append('lemmatize_diff')

    feature_dict = {}
    for key in array_keys:
      if key == 'indices':
        dtype = tf.int64
      else:
        dtype = tf.string

      if key == 'tokenize_diff':
        shape = [num_cols]
      else:
        shape = [num_cols * self.max_sent_len]
      feature_dict[key] = tf.FixedLenFeature(shape, dtype)
    return feature_dict

  def __len__(self):
    """Get the length of the vector outputted by the row_to_vector method."""
    return self.max_sent_len
