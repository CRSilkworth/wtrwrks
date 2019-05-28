"""DocumentTransform definition."""
import transform as n
import numpy as np
import wtrwrks.waterworks.name_space as ns
import wtrwrks.tanks.tank_defs as td
import wtrwrks.read_write.tf_features as feat
import wtrwrks.transforms.string_transform as st
from wtrwrks.waterworks.empty import empty
import os
import tensorflow as tf
import pprint
import hashlib

class DocumentToSentenceTransform(n.Transform):
  """Class used to transform entire documents into standardized vectors along with any supplemental data needed in order to reconstruct the original documents.

  Parameters
  ----------
  name : str
    The name of the transform.
  from_file : str
    The path to the saved file to recreate the transform object that was saved to disk.
  save_dict : dict
    The dictionary to recreate the transform object
  sent_tokenizer : func
    A function that takes in a string and splits it up into a list of sentences.
  sent_detokenizer : func
    A function that takes in a list of sentences and outputs a string. Doesn't have to be an exact inverse to sent_tokenizer but should be close otherwise a lot of large diff strings will have to be outputted in order to reproduce the original strings.
  string_transform : StringTransform
    The transform that describes how to transform individual sentences into standardized vectors an back.
  keep_dims : bool
    Whether or not to keep the original dimensions of the inputted array. If the dimensions are kept then an array is produced with an extract dimension which is to be interpretted as the document sentences dimension. The size of this dimension is set by max_doc_len. If the dimension aren't kept the outputted array will have a row for each sentence. This means the number of examples that will be produced from a collection of documents will be the sum of number of sentences in each document.
  Attributes
  ----------
  input_dtype: numpy dtype
    The datatype of the original inputted array.
  input_shape: list of ints
    The shape of the original inputted array.

  """
  attribute_dict = {'name': '', 'dtype': np.int64, 'input_dtype': None, 'sent_tokenizer': None, 'sent_detokenizer': lambda a: ''.join(a), 'max_doc_len': None, 'keep_dims': True}

  def __len__(self):
    """Get the length of the vector outputted by the row_to_vector method."""
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
    # Unique document ids are needed for the case that all the sentences are
    # put on the same axis (i.e. keep_dims=False) because there is no way to
    # distinguish which sentences came from which documents.
    r_dict = {}
    r_dict[self._pre('sentences', prefix)] = tap_dict[self._pre('sentences', prefix)]
    if not self.keep_dims:
      r_dict[self._pre('ids', prefix)] = tap_dict[self._pre('ids', prefix)]

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
    funnel_dict = {}
    if array is not None:
      funnel_dict['input'] = array

    if array is not None and not self.keep_dims:
      num_pours = np.full(array.shape, self.num_pours),
      indices = np.arange(array.size).reshape(array.shape)

      # A simple hash function is used to create the document ids. The ids are
      # built from the location within the original array the document appears
      # and the number of times that the pour method is called.
      v_hash = np.vectorize(tup_hash)
      funnel_dict['doc_ids'] = v_hash(num_pours, indices)
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
    tap_dict = {}
    tap_dict[self._pre('sentences', prefix)] = pour_outputs[self._pre('sentences', prefix)]
    if not self.keep_dims:
      tap_dict[self._pre('FlatTokenize_0/tubes/shape', prefix)] = self.input_shape
      tap_dict[self._pre('ids', prefix)] = pour_outputs[self._pre('ids', prefix)]

    return tap_dict

  def _parse_examples(self, arrays_dict, prefix=''):
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
    pour_outputs = {}
    pour_outputs.update(arrays_dict)

    if not self.keep_dims:
      pour_outputs[self._pre('ids', prefix)] = arrays_dict[self._pre('ids', prefix)].reshape([-1])

    return pour_outputs

  def _setattributes(self, **kwargs):
    """Set the actual attributes of the Transform and do some value checks to make sure they valid inputs.

    Parameters
    ----------
    **kwargs :
      The keyword arguments that set the values of the attributes defined in the attribute_dict.

    """
    super(DocumentToSentenceTransform, self)._setattributes(**kwargs)

    if self.sent_tokenizer is None:
      raise ValueError("Must specify a sentence tokenizer (sent_tokenizer). Got " + str(self.sent_tokenizer))

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
    att_dict = {}
    shape = list(self.input_shape[1:]) + [self.max_doc_len]
    att_dict['sentences'] = {
      'shape': shape if self.keep_dims else [1],
      'tf_type': tf.string,
      'size': feat.size_from_shape(shape) if self.keep_dims else 1,
      'feature_func': feat._bytes_feat,
      'np_type': np.unicode
    }
    if not self.keep_dims:
      att_dict['ids'] = {
        'shape': [],
        'tf_type': tf.string,
        'size': 1,
        'feature_func': feat._bytes_feat,
        'np_type': np.unicode
      }
    att_dict = self._pre(att_dict, prefix)
    return att_dict

  def calc_global_values(self, documents, verbose=True):
    """Calculate all the values of the Transform that are dependent on all the examples of the dataset. (e.g. mean, standard deviation, unique category values, etc.) This method must be run before any actual transformation can be done.

    Parameters
    ----------
    array : np.ndarray
      The entire dataset.
    verbose : bool
      Whether or not to print out warnings.

    """
    self.input_dtype = documents.dtype
    self.input_shape = documents.shape
    self.num_pours = 0

    all_sentences = []
    max_len = None
    # Tokenize each document into sentences and get the max document length.
    for document in documents.flatten():
      sentences = self.sent_tokenizer(document)
      all_sentences.append(sentences)
      cur_len = len(sentences)
      if max_len is None or max_len < cur_len:
        max_len = cur_len

    if self.max_doc_len is None:
      self.max_doc_len = max_len

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
    # If the dimensions are not being kept then run flat_tokenize which puts
    # all sentences on the same axis.
    if not self.keep_dims:
      sents, sents_slots = td.flat_tokenize(strings=array, tokenizer=self.sent_tokenizer, detokenizer=self.sent_detokenizer)
      sents_slots['ids'].set_name('doc_ids')
      sents['ids'].set_name('ids')

    else:
      sents, sents_slots = td.tokenize(strings=array, tokenizer=self.sent_tokenizer, detokenizer=self.sent_detokenizer, max_len=self.max_doc_len)

    sents_slots['strings'].set_name('input')
    sents['target'].set_name('sentences')

    if return_tubes is not None:
      ww = sents['target'].waterwork
      r_tubes = []
      for r_tube_key in return_tubes:
        r_tubes.append(ww.maybe_get_tube(r_tube_key))
      return r_tubes

def tup_hash(*args):
  return hashlib.sha224(str(args)).hexdigest()
