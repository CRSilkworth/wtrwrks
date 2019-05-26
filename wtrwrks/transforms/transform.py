"""Definition of the base Transform class"""
import pandas as pd
import wtrwrks.utils.dir_functions as d
import wtrwrks.waterworks.waterwork as wa
import os
import numpy as np
import tensorflow as tf


class Transform(object):
  """Abstract class used to create mappings from raw to vectorized, normalized data and vice versa. These transform store all the information necessary to create a Waterwork object which can do all the necessary reversible alterations to the data and also to transform it back to its original form.

  Parameters
  ----------
  from_file : str
    The path to the saved file to recreate the transform object that was saved to disk.
  save_dict : dict
    The dictionary to recreate the transform object

  Attributes
  ----------
  attribute_dict: dict
    The keys are the attributes of the class while the values are the default values. It's done this way rather than defined in the __init__ because this dictionary also defines what values need to be saved when written to disk, and what values need to be displayed when printed to the terminal.

  """

  attribute_dict = {'name': ''}

  def __init__(self, from_file=None, save_dict=None, **kwargs):
    """Define a transform using a dictionary, file, or by setting the attribute values in kwargs

    Parameters
    ----------
    from_file : None or str
        The file path of the Tranform that was written to disk.
    save_dict : dict or None
        The dictionary of attributes that completely define a Transform.
    **kwargs :
        The keyword arguments that set the values of the attributes defined in the attribute_dict.

    """
    if from_file is not None:
      save_dict = d.read_from_file(from_file)
      self._from_save_dict(save_dict)
    elif save_dict is not None:
      self._from_save_dict(save_dict)
    else:
      self._setattributes(**kwargs)

    self.waterwork = None

  def __len__(self):
    """Get the length of the vector outputted by the row_to_vector method."""
    return len(getattr(self, self.attribute_list[0]))

  def __str__(self):
    """Return the stringified values for each of the attributes in attribute list."""
    return str({a: str(getattr(self, a)) for a in self.attribute_dict})

  def _extract_pour_outputs(self, tap_dict, prefix=''):
    raise NotImplementedError()

  def _extract_pump_outputs(self, funnel_dict, prefix=''):
    raise NotImplementedError()

  def _feature_def(self, num_cols=1, prefix=''):
    raise NotImplementedError()

  def _from_save_dict(self, save_dict):
    """Reconstruct the transform object from the dictionary of attributes."""
    for key in self.attribute_dict:
      setattr(self, key, save_dict[key])

  def _full_missing_vals(self, mask, missing_vals):
    """Create an array of the same shape as mask, with all default values except for those that will be filled by missing_vals.

    Parameters
    ----------
    mask : np.ndarray of bools
      A boolean mask of some shape.
    missing_vals : np.ndarray
      A one dimensional array of values to fill in the non default values of the returned array. The number of mssing values must equal the number of Trues that appear in 'mask'.

    Returns
    -------
    np.ndarray of the same dtype as missing_vals
      An array of all default values with the same shape as 'mask' but with the elements of missing_vals in the place of the Trues from 'mask'.

    """
    dtype = self.input_dtype

    # Choose different default values, and type for the returned array depending
    # on the input_dtype
    if dtype.type in (np.string_, np.unicode_):
      str_len = max([len(i) for i in missing_vals] + [1])
      full_missing_vals = np.full(mask.shape, '', dtype='|U' + str(str_len))
    elif dtype in (np.int64, np.int32, np.float64, np.float32):
      full_missing_vals = np.zeros(mask.shape, dtype=dtype)
    else:
      raise TypeError("Only string and number types are supported. Got " + str(dtype))

    full_missing_vals[mask] = missing_vals
    return full_missing_vals

  def _get_example_dicts(self, pour_outputs, prefix=''):
    raise NotImplementedError()

  def _get_funnel_dict(self, array=None, prefix=''):
    raise NotImplementedError()

  def _get_tap_dict(self, pour_outputs, prefix=''):
    raise NotImplementedError()

  def _nopre(self, d, prefix=''):
    """Strip the self.name/prefix from a string or keys of a dictionary.

    Parameters
    ----------
    d : str or dict
      The string or dictionary to strip the prefix/self.name  from.
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    str or dict
      The string with the prefix/self.name stripped or the dictionary with the prefix/self.name stripped from all the keys.

    """
    str_len = len(os.path.join(prefix, self.name) + '/')
    if str_len == 1:
      str_len = 0
    if type(d) is not dict:
      return d[str_len:]

    r_d = {}
    for key in d:
      if type(key) is tuple and type(key[0]) in (str, unicode):
        r_d[(key[0][str_len:], key[1])] = d[key]
      elif type(key) in (str, unicode):
        r_d[key[str_len:]] = d[key]
      else:
        r_d[key] = d[key]
    return r_d

  def _parse_example_dicts(self, example_dicts, prefix=''):
    raise NotImplementedError()

  def _pre(self, d, prefix=''):
    """Add the name and some additional prefix to the keys in a dictionary or to a string directly

    Parameters
    ----------
    d : str or dict
      The string or dictionary to add to the prefix/self.name prefix to.
    prefix : str
      Any additional prefix to give the string/dictionary keys. Defaults to no additional prefix.

    Returns
    -------
    str or dict
      The string with the prefix/self.name added or the dictionary with the prefix/self.name added to all the keys.

    """
    if type(d) is not dict:
      return os.path.join(prefix, self.name, d)

    r_d = {}
    for key in d:
      if type(key) is tuple and type(key[0]) in (str, unicode):
        r_d[(os.path.join(prefix, self.name, key[0]), key[1])] = d[key]
      elif type(key) in (str, unicode):
        r_d[os.path.join(prefix, self.name, key)] = d[key]
      else:
        r_d[key] = d[key]
    return r_d

  def _save_dict(self):
    """Create the dictionary of values needed in order to reconstruct the transform."""
    save_dict = {}
    for key in self.attribute_dict:
      save_dict[key] = getattr(self, key)
    return save_dict

  def _setattributes(self, **kwargs):
    """Set the actual attributes of the Transform and do some value checks to make sure they valid inputs.

    Parameters
    ----------
    **kwargs :
      The keyword arguments that set the values of the attributes defined in the attribute_dict.

    """
    attribute_set = set(self.attribute_dict)
    invalid_keys = sorted(set(kwargs.keys()) - attribute_set)

    if invalid_keys:
      raise ValueError("Keyword arguments: " + str(invalid_keys) + " are invalid.")

    for key in self.attribute_dict:
      if key in kwargs:
        setattr(self, key, kwargs[key])
      else:
        setattr(self, key, self.attribute_dict[key])

    if '/' in self.name:
      raise ValueError("Cannot give Transform a name with '/'. Got " + str(self.name))

  def _shape_def(self, prefix=''):
    raise NotImplementedError()

  def define_waterwork(self):
    raise NotImplementedError()

  def get_waterwork(self, recreate=False):
    """Create the Transform's waterwork or return the one that was already created.

    Parameters
    ----------
    recreate : bool
      Whether or not to force the transform to create a new waterwork.

    Returns
    -------
    Waterwork
      The waterwork object that this transform creates.

    """

    assert self.input_dtype is not None, ("Run calc_global_values before running the transform")

    if self.waterwork is not None and not recreate:
      return self.waterwork

    with wa.Waterwork(name=self.name) as ww:
      self.define_waterwork()

    self.waterwork = ww
    return ww

  def pour(self, array):
    """Execute the transformation in the pour (forward) direction.

    Parameters
    ----------
    array : np.ndarray
      The numpy array to transform.

    Returns
    -------
    dict
      The dictionary of transformed outputs as well as any additional information needed to completely reconstruct the original rate.

    """
    ww = self.get_waterwork()
    funnel_dict = self._get_funnel_dict(array)
    tap_dict = ww.pour(funnel_dict, key_type='str')
    return self._extract_pour_outputs(tap_dict)

  def pour_examples(self, array):
    """Run the pour transformation on an array to transform it into a form best for ML pipelines. This list of example dictionaries can be easily converted into tf records, but also have all the information needed in order to reconstruct the original array.

    Parameters
    ----------
    array : np.ndarray
      The numpy array to transform into examples.

    Returns
    -------
    list of dicts of features
      The example dictionaries which contain tf.train.Features.

    """
    pour_outputs = self.pour(array)
    example_dicts = self._get_example_dicts(pour_outputs)
    return example_dicts

  def pump(self, kwargs):
    """Execute the transformation in the pump (backward) direction.

    Parameters
    ----------
    kwargs: dict
      The dictionary all information needed to completely reconstruct the original rate.

    Returns
    -------
    array : np.ndarray
      The original numpy array that was poured.

    """
    ww = self.get_waterwork()
    tap_dict = self._get_tap_dict(kwargs)
    funnel_dict = ww.pump(tap_dict, key_type='str')
    return self._extract_pump_outputs(funnel_dict)

  def pump_examples(self, example_dicts):
    """Run the pump transformation on a list of example dictionaries to reconstruct the original array.

    Parameters
    ----------
    example_dicts: list of dicts of arrays
      The example dictionaries which the arrays associated with a single example.

    Returns
    -------
    np.ndarray
      The numpy array to transform into examples.

    """
    pour_outputs = self._parse_example_dicts(example_dicts)
    return self.pump(pour_outputs)

  def read_and_decode(self, serialized_example, prefix=''):
    """Convert a serialized example created from an example dictionary from this transform into a dictionary of shaped tensors for a tensorflow pipeline.

    Parameters
    ----------
    serialized_example : tfrecord serialized example
      The serialized example to read and convert to a dictionary of tensors.
    prefix : str
      A string to prefix the dictionary keys with.

    Returns
    -------
    dict of tensors
      The tensors created by decoding the serialized example

    """
    feature_dict = self._feature_def(prefix)
    shape_dict = self._shape_def(prefix)

    features = tf.parse_single_example(
      serialized_example,
      features=feature_dict
    )

    for key in shape_dict:
      features[key] = tf.reshape(features[key], shape_dict[key])

    return features

  def save_to_file(self, path):
    """Save the transform object to disk."""
    save_dict = self._save_dict()
    d.save_to_file(save_dict, path)

  def write_examples(self, array, file_name):
    """Pours the array then writes the examples to tfrecords.

    Parameters
    ----------
    array : np.ndarray
      The array to transform to examples, then write to disk.
    file_name : str
      The name of the tfrecord file to write to.

    """
    example_dicts = self.pour_examples(array)
    writer = tf.python_io.TFRecordWriter(file_name)

    for feature_dict in example_dicts:
      example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict)
      )
      writer.write(example.SerializeToString())

    writer.close()
