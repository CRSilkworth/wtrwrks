import transform as n
import numpy as np
import reversible_transforms.waterworks.name_space as ns
import reversible_transforms.tanks.tank_defs as td
import reversible_transforms.read_write.tf_features as feat
import reversible_transforms.transforms.string_transform as st
from reversible_transforms.waterworks.empty import empty
import os
import tensorflow as tf
import pprint

class DocumentTransform(n.Transform):
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
  attribute_dict = {'name': '', 'dtype': np.int64, 'input_dtype': None, 'sent_tokenizer': None, 'sent_detokenizer': lambda a: ''.join(a), 'string_transform': None, 'max_doc_len': None}

  def _setattributes(self, **kwargs):
    super(DocumentTransform, self)._setattributes(**kwargs)

    if not isinstance(self.string_transform, st.StringTransform):
      raise ValueError("Must supply a StringTransform to string_transform. Got " + type(self.string_transform))

    if not self.string_transform.name:
      raise ValueError("string_transform must have name. Got " + str(self.string_transform.name))

    if self.sent_tokenizer is None:
      raise ValueError("Must specify a sentence tokenizer (sent_tokenizer). Got " + str(self.sent_tokenizer))

  def calc_global_values(self, documents, verbose=True):
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
    self.input_dtype = documents.dtype
    if len(documents.shape) != 2:
      raise ValueError("Must supply an array of rank 2 which represent axes of (examples, columns). Got " + str(len(documents.shape)))
    array = []

    document_lengths = []
    all_sentences = []
    max_len = None
    for cols in documents:

      document_sentences = []
      for document in cols:
        sentences = self.sent_tokenizer(document)
        document_sentences.append(sentences)

        cur_len = len(sentences)
        if max_len is None or max_len < cur_len:
          max_len = cur_len

      all_sentences.append(document_sentences)

    if self.max_doc_len is None:
      self.max_doc_len = max_len

    array = np.empty(list(documents.shape) + [self.max_doc_len], dtype=self.input_dtype)
    for row_num, document_sentences in enumerate(all_sentences):
      for col_num, sentences in enumerate(document_sentences):
        for sent_num, sentence in enumerate(sentences):
          array[row_num, col_num, sent_num] = sentence

    self.string_transform.calc_global_values(array)

  def define_waterwork(self, array=empty):
    sents, sents_slots = td.tokenize(strings=array, tokenizer=self.sent_tokenizer, detokenizer=self.sent_detokenizer, max_len=self.max_doc_len)
    sents_slots['strings'].set_name('input')

    with ns.NameSpace(self.string_transform.name):
      self.string_transform.define_waterwork(array=sents['target'])

  def _get_funnel_dict(self, array=None, prefix=''):
    funnel_dict = self.string_transform._get_funnel_dict()
    if array is not None:
      funnel_dict['input'] = array

    return self._pre(funnel_dict, prefix)

  def _extract_pour_outputs(self, tap_dict, prefix=''):
    r_dict = self.string_transform._extract_pour_outputs(tap_dict, os.path.join(prefix, self.name))
    return r_dict

  def _get_tap_dict(self, pour_outputs, prefix=''):
    tap_dict = self.string_transform._get_tap_dict(pour_outputs, os.path.join(prefix, self.name))
    return tap_dict

  def _extract_pump_outputs(self, funnel_dict, prefix=''):
    return funnel_dict[self._pre('input', prefix)]

  def _get_example_dicts(self, pour_outputs, prefix=''):
    pour_outputs = self._nopre(pour_outputs, prefix)
    mask = pour_outputs['indices'] == self.unk_index
    missing_vals = pour_outputs['missing_vals']

    pour_outputs['missing_vals'] = self._full_missing_vals(mask, missing_vals)

    array_keys = ['indices', 'tokenize_diff', 'missing_vals']
    if self.lower_case:
      array_keys.append(self._pre('lower_case_diff'))
    if self.half_width:
      array_keys.append(self._pre('half_width_diff'))
    if self.lemmatize:
      array_keys.append(self._pre('lemmatize_diff'))

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
          fixed_array = fixed_array.reshape([-1, self.max_sent_len])

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
      array_keys.append(self._pre('lower_case_diff', prefix))
    if self.half_width:
      array_keys.append(self._pre('half_width_diff', prefix))
    if self.lemmatize:
      array_keys.append(self._pre('lemmatize_diff', prefix))

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

    feature_dict = self._pre(feature_dict, prefix)
    return feature_dict

  def __len__(self):
    """Get the length of the vector outputted by the row_to_vector method."""
    return self.max_sent_len
