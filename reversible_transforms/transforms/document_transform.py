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
import hashlib

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
  attribute_dict = {'name': '', 'dtype': np.int64, 'input_dtype': None, 'sent_tokenizer': None, 'sent_detokenizer': lambda a: ''.join(a), 'string_transform': None, 'max_doc_len': None, 'keep_dims': True}

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
    self.input_shape = documents.shape
    self.num_pours = 0

    # if len(documents.shape) != 2:
    #   raise ValueError("Must supply an array of rank 2 which represent axes of (examples, columns). Got " + str(len(documents.shape)))
    array = []
    all_sentences = []
    max_len = None
    for document in documents.flatten():
      sentences = self.sent_tokenizer(document)
      all_sentences.append(sentences)
      cur_len = len(sentences)
      if max_len is None or max_len < cur_len:
        max_len = cur_len

    if self.max_doc_len is None:
      self.max_doc_len = max_len

    if self.keep_dims:
      stand_sentences = []
      for document, sentences in zip(documents.flatten(), all_sentences):
        doc_diff = self.max_doc_len - len(sentences)
        if doc_diff >= 0:
          sentences = np.pad(sentences, (0, doc_diff), 'constant', constant_values=('', '')).astype(self.input_dtype)
        else:
          sentences = np.array(sentences[:doc_diff]).astype(self.input_dtype)
        stand_sentences.append(sentences)

      array = np.concatenate(stand_sentences)
      array = array.reshape(list(self.input_shape) + [self.max_doc_len])

    else:
      array = np.concatenate(all_sentences).astype(self.input_dtype)

    self.string_transform.calc_global_values(array)

  def define_waterwork(self, array=empty):
    if not self.keep_dims:
      sents, sents_slots = td.flat_tokenize(strings=array, tokenizer=self.sent_tokenizer, detokenizer=self.sent_detokenizer)
      sents_slots['ids'].set_name('doc_ids')
      sents['ids'].set_name('ids')
    else:
      sents, sents_slots = td.tokenize(strings=array, tokenizer=self.sent_tokenizer, detokenizer=self.sent_detokenizer, max_len=self.max_doc_len)

    sents_slots['strings'].set_name('input')

    with ns.NameSpace(self.string_transform.name):
      self.string_transform.define_waterwork(array=sents['target'])

  def _get_funnel_dict(self, array=None, prefix=''):
    funnel_dict = self.string_transform._get_funnel_dict()
    if array is not None:
      funnel_dict['input'] = array

    if array is not None and not self.keep_dims:
      num_pours = np.full(array.shape, self.num_pours),
      indices = np.arange(array.size).reshape(array.shape)

      v_hash = np.vectorize(tup_hash)
      funnel_dict['doc_ids'] = v_hash(num_pours, indices)
    return self._pre(funnel_dict, prefix)

  def _extract_pour_outputs(self, tap_dict, prefix=''):
    r_dict = self.string_transform._extract_pour_outputs(tap_dict, os.path.join(prefix, self.name))
    if not self.keep_dims:
      r_dict[self._pre('ids', prefix)] = tap_dict[self._pre('ids', prefix)]
    return r_dict

  def _get_tap_dict(self, pour_outputs, prefix=''):
    tap_dict = self.string_transform._get_tap_dict(pour_outputs, os.path.join(prefix, self.name))
    if not self.keep_dims:
      tap_dict[self._pre('FlatTokenize_0/tubes/shape', prefix)] = self.input_shape
      tap_dict[self._pre('ids', prefix)] = pour_outputs[self._pre('ids', prefix)]

    return tap_dict

  def _extract_pump_outputs(self, funnel_dict, prefix=''):
    return funnel_dict[self._pre('input', prefix)]

  def _get_example_dicts(self, pour_outputs, prefix=''):
    example_dicts = self.string_transform._get_example_dicts(pour_outputs, os.path.join(prefix, self.name))

    if not self.keep_dims:
      for row_num, example_dict in enumerate(example_dicts):
        rs = pour_outputs[self._pre('ids', prefix)][row_num]
        example_dict[self._pre('ids', prefix)] = feat._bytes_feat(rs)

    return example_dicts

  def _parse_example_dicts(self, example_dicts, prefix=''):
    pour_outputs = self.string_transform._parse_example_dicts(example_dicts, os.path.join(prefix, self.name))

    if not self.keep_dims:
      ids = []
      for example_dict in example_dicts:
        r_id = example_dict[self._pre('ids', prefix)]
        ids.append(r_id)
      pour_outputs[self._pre('ids', prefix)] = np.concatenate(ids)

    return pour_outputs

  def _feature_def(self, num_cols=1, prefix=''):
    num_cols = 1
    if self.keep_dims:
      num_cols = np.prod(self.input_shape[1:]) * self.max_doc_len

    feature_dict = self.string_transform._feature_def(prefix=os.path.join(prefix, self.name))
    if not self.keep_dims:
      feature_dict[self._pre('ids', prefix)] = tf.FixedLenFeature([1], tf.string)
    return feature_dict

  def _shape_def(self, prefix=''):
    shape_dict = self.string_transform._shape_def(os.path.join(prefix, self.name))
    if not self.keep_dims:
      shape_dict[self._pre('ids', prefix)] = ()
    return shape_dict

  def __len__(self):
    """Get the length of the vector outputted by the row_to_vector method."""
    return self.max_sent_len


def tup_hash(*args):
  return hashlib.sha224(str(args)).hexdigest()
