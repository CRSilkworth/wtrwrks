"""Definition of the base Transform class"""
import pandas as pd
import wtrwrks.utils.dir_functions as d
import wtrwrks.tanks.utils as ut
import wtrwrks.utils.array_functions as af
import wtrwrks.waterworks.waterwork as wa
import wtrwrks.read_write.tf_features as feat
import os
import numpy as np
import tensorflow as tf
import itertools
import jpype
import wtrwrks.utils.multiprocessing as mh
import wtrwrks.utils.batch_functions as b
import logging
import re
import inspect
import random
import glob


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

  attribute_dict = {'name': '', 'cols': None, 'num_examples': None, 'is_calc_run': False, 'input_dtype': None, 'dtype': None}
  required_params = set(['name'])

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
      attribute_set = set(self.attribute_dict)
      invalid_keys = sorted(set(kwargs.keys()) - attribute_set)

      if invalid_keys:
        raise TypeError("{} got unexpected keyword argument {}".format(self.__class__.__name__, invalid_keys[0]))

      for key in self.attribute_dict:
        if key in kwargs:
          setattr(self, key, kwargs[key])
        else:
          setattr(self, key, self.attribute_dict[key])

      for key in sorted(self.required_params):
        if key not in kwargs:
          raise TypeError("Must supply '{}' as an argument".format(key))

    self.waterwork = None

  def __str__(self):
    """Return the stringified values for each of the attributes in attribute list."""
    return str({a: str(getattr(self, a)) for a in self.attribute_dict})

  def __len(self):
    raise NotImplementedError()

  def _array_dicts_to_array_dict(self, array_dicts):
    """Convert the list of array dictionaries into a dictionary of arrays, by stacking across all dictionaries in the list.

    Parameters
    ----------
    array_dicts : list of dicts
      The list of array dictionaries to convert into a single array dictionary.

    Returns
    -------
    dict of numpy array
      All the arrays from the lists stacked along the 0th dimension.

    """

    r_dict = {}
    for array_dict in array_dicts:
      for key in array_dict:
        r_dict.setdefault(key, [])
        r_dict[key].append(array_dict[key])

    for key in r_dict:
      r_dict[key] = np.stack(r_dict[key])
    return r_dict

  def _from_save_dict(self, save_dict):
    """Reconstruct the transform object from the dictionary of attributes."""
    for key in self.attribute_dict:
      setattr(self, key, save_dict[key])

  def _get_array_attributes(self, prefix):
    raise NotImplementedError()

  def _get_dataset(self, file_name_pattern, batch_size, num_epochs=None, num_steps=None, filters=None, keep_features=None, drop_features=None, add_tensors=None, num_threads=1, shuffle_buffer_size=10000, random_seed=None):
    random.seed(random_seed)
    file_names = glob.glob(file_name_pattern)
    file_names.sort()

    shuffled_file_names = random.sample(file_names, len(file_names))

    dataset = tf.data.TFRecordDataset(shuffled_file_names)
    if num_steps is not None:
      dataset = dataset.take(num_steps)
      dataset = dataset.shuffle(
          buffer_size=shuffle_buffer_size
      )
    else:
      if num_epochs is not None and type(num_epochs) is not int and not (tf.contrib.framework.is_tensor(num_epochs) and num_epochs.dtype is tf.int64):
        logging.warn('%s is not a whole number. Will be converted to %s', num_epochs, int(num_epochs))

        num_epochs = int(num_epochs)

      s_and_r = tf.data.experimental.shuffle_and_repeat(
          buffer_size=shuffle_buffer_size,
          count=num_epochs
      )
      dataset = dataset.apply(s_and_r)

    # Read and decode the dataset
    dataset = dataset.map(
        lambda se: self.read_and_decode(se, self.name, keep_features, drop_features),
        num_parallel_calls=num_threads
    )

    # If any filters were put in place filter the values.
    if filters is not None:
      for key in filters:
        dataset = dataset.filter(
          filters[key]
        )

    if batch_size is not None:
      dataset = dataset.batch(batch_size)

    if add_tensors is not None:
      for key in add_tensors:
        def _add_tensor(kwargs):
          kwargs[key] = add_tensors[key]
          return kwargs
        dataset = dataset.map(_add_tensor)
    return dataset

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

  def _maybe_convert_to_df(self, data):
    if type(data) is pd.DataFrame:
      return data
    else:
      return pd.DataFrame(
        data=data,
        columns=self.cols,
        index=np.arange(data.shape[0])
      )

  def _parse_examples(self, arrays_dict, prefix=''):
    return arrays_dict

  def _pre(self, d, prefix=''):
    """Add the name and some additional prefix to the keys in a dictionary or to a string directly.

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
      obj = getattr(self, key)
      save_dict[key] = obj
    save_dict['__class__'] = str(self.__class__.__name__)
    save_dict['__module__'] = str(self.__class__.__module__)
    return save_dict

  def _start_calc(self):
    """Default function to run at the start of calc_global_values"""
    self.num_examples = 0.

  def _finish_calc(self):
    """Default function to run at the end of calc_global_values"""
    return

  def calc_global_values(self, data=None, data_iter=None):
    if data is not None and data_iter is None:
      data_iter = [data]
    elif data_iter is not None and data is None:
      pass
    else:
      raise ValueError("Must supply exactly one array or array_iter.")

    self._start_calc()
    has_at_least_one = False
    for data_num, data in enumerate(data_iter):
      if data_num == 0:
        has_at_least_one = True
        if type(data) is pd.DataFrame:
          is_df = True
          self.cols = list(data.columns)
        else:
          is_df = False
          self.cols = [self.name + '_' + str(dim) for dim in xrange(data.shape[1])]
      data = data.values if is_df else data

      if data_num == 0:
        if self.input_dtype is None:
          self.input_dtype = data.dtype

        if len(data.shape) != 2:
          raise ValueError("Only rank 2 arrays are supported for transforms. Got {}".format(len(data.shape)))

      self._calc_global_values(data)
    self._finish_calc()

    if not has_at_least_one:
      raise ValueError("No data was passed to calc_global_values.")
    self.is_calc_run = True

  def define_waterwork(self, array=None, return_tubes=None):
    raise NotImplementedError()

  def get_default_array(self, batch_size=1):
    #####################################
    # NOTE: PUT BACK IN CHECK AFTER WORKFLOW IS RERUN
    # assert self.is_calc_run, ("Run calc_global_values before getting default array")
    ####################################
    return af.empty_array([batch_size, len(self.cols)], self.input_dtype)

  def get_dataset_iter(self, file_name_pattern, batch_size, keep_features=None, drop_features=None, add_tensors=None):

    dataset = self._get_dataset(file_name_pattern, batch_size, keep_features=keep_features, drop_features=drop_features, add_tensors=add_tensors)

    data_iter = tf.data.Iterator.from_structure(
        dataset.output_types,
        dataset.output_shapes
    )

    return data_iter

  def get_dataset_feed_iter(self, file_name_pattern, batch_size, keep_features=None, drop_features=None, add_tensors=None):

    dataset = self._get_dataset(file_name_pattern, batch_size, keep_features=keep_features, drop_features=drop_features, add_tensors=add_tensors)
    handle = tf.placeholder(tf.string, shape=[])

    feed_iter = tf.data.Iterator.from_string_handle(
        handle,
        dataset.output_types,
        dataset.output_shapes
    )

    return feed_iter, handle

  def get_dataset_iter_init(self, dataset_iter, file_name_pattern, batch_size, num_epochs=None, num_steps=None, filters=None, keep_features=None, drop_features=None, add_tensors=None, num_threads=1, shuffle_buffer_size=1000, random_seed=None):
    dataset = self._get_dataset(file_name_pattern, batch_size, num_epochs, num_steps, filters, keep_features, drop_features, add_tensors, num_threads, shuffle_buffer_size, random_seed)

    return dataset_iter.make_initializer(dataset)

  def get_placeholder(self, tap_key, with_batch=True, batch=None):
    att_dict = self._get_array_attributes()

    if with_batch:
      shape = [batch] + att_dict[tap_key]['shape']
    else:
      shape = att_dict[tap_key]['shape']

    if att_dict[tap_key]['np_type'] == np.float64:
      dtype = tf.float64
    else:
      dtype = att_dict[tap_key]['tf_type']

    ph = tf.placeholder(
      dtype=dtype,
      shape=shape
    )

    return ph

  def get_schema_dict(self, var_lim=255):

    if self.input_dtype in (np.int64, np.int32, int):
      db_dtype = 'INTEGER'
    elif self.input_dtype in (np.float64, np.float32, float):
      db_dtype = 'FLOAT'
    elif self.input_dtype in (np.bool, bool):
      db_dtype = 'BOOLEAN'
    elif self.input_dtype in (np.dtype('S'), np.dtype('U'), np.dtype('O')):
      db_dtype = 'VARCHAR'
    elif self.input_dtype in (np.datetime64,):
      db_dtype = 'TIMESTAMP'
    else:
      raise ValueError("{} is not a supported type to be used on the database.".format())

    schema_dict = {}
    for col in self.cols:
      if db_dtype == 'VARCHAR' and type(var_lim) is dict and col in var_lim:
        schema_dict[col] = db_dtype + '(' + str(var_lim[col]) + ')'
      elif db_dtype == 'VARCHAR' and type(var_lim) is dict:
         schema_dict[col] = db_dtype + '(' + str(255) + ')'
      elif db_dtype == 'VARCHAR':
        schema_dict[col] = db_dtype + '(' + str(var_lim) + ')'
      else:
        schema_dict[col] = db_dtype
    return schema_dict

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
    assert self.is_calc_run, ("Run calc_global_values before running the transform")

    if self.waterwork is not None and not recreate:
      return self.waterwork

    with wa.Waterwork(name=self.name) as ww:
      self.define_waterwork()

    self.waterwork = ww
    return ww

  def write_examples(self, data=None, data_iter=None, file_name=None, file_num_offset=0, batch_size=1, num_threads=1, skip_fails=False, skip_keys=None, use_threading=False, serialize_func=None, prefix=''):
    """Pours the arrays then writes the examples to tfrecords in a multithreading manner. It creates one example per 'row', i.e. axis=0 of the arrays. All arrays must have the same axis=0 dimension and must be of a type that can be written to a tfrecord

    Parameters
    ----------
    funnel_dicts : list of dicts(
      keys - Slot objects or Placeholder objects. The 'funnels' (i.e. unconnected slots) of the waterwork.
      values - valid input data types
    )
      The inputs to the waterwork's full pour functions. There is exactly one funnel_dict for every process.
    file_name : str
      The name of the tfrecord file to write to. An extra '_<num>' will be added to the name.
    file_num_offset : int
      A number that controls what number will be appended to the file name (so that files aren't overwritten.)

    """
    if data is not None and data_iter is None:
      data_iter = [data]
    elif data_iter is not None and data is None:
      pass
    else:
      raise ValueError("Must supply exactly one data or data_iter.")

    if serialize_func is None:
      save_dict = self._save_dict()
      cls = self.__class__

      def serialize_func(data):
        trans = cls(save_dict=save_dict)

        tap_dict = trans.pour(data)
        example_dicts = trans.tap_dict_to_examples(tap_dict, prefix)

        serials = []
        for example_dict in example_dicts:
          example = tf.train.Example(
            features=tf.train.Features(feature=example_dict)
          )
          serials.append(example.SerializeToString())

        return serials

    if type(data_iter) in (list, tuple):
      data_iter = (i for i in data_iter)

    file_names = []
    if not file_name.endswith('.tfrecord'):
      raise ValueError("file_name must end in '.tfrecord'")

    dir = '/'.join(file_name.split('/')[:-1])
    d.maybe_create_dir(dir)

    for batch_num, batch in enumerate(b.batcher(data_iter, batch_size)):
      logging.info("Serializing batch %s", batch_num)
      if skip_fails:
        try:
          all_serials = mh.multi_map(serialize_func, batch, num_threads, use_threading)
        except Exception:
          logging.warn("Batched %s failed. Skipping.", batch_num)
          continue
      else:
        all_serials = mh.multi_map(serialize_func, batch, num_threads, use_threading)

      logging.info("Finished serializing batch %s", batch_num)

      file_num = file_num_offset + batch_num
      fn = file_name.replace('.tfrecord', '_' + str(file_num) + '.tfrecord')
      file_names.append(fn)

      logging.info("Writing batch %s", batch_num)
      writer = tf.python_io.TFRecordWriter(fn)
      for serials in all_serials:
        for serial in serials:
          writer.write(serial)

      logging.info("Finished writing batch %s", batch_num)
      writer.close()

    return file_names

  def pour(self, data=None, data_iter=None):
    """Execute the transformation in the pour (forward) direction.

    Parameters
    ----------
    data : np.ndarray
      The numpy array to transform.

    Returns
    -------
    dict
      The dictionary of transformed outputs as well as any additional information needed to completely reconstruct the original rate.

    """
    ww = self.get_waterwork()

    def normalize_and_pour(data):
      if type(data) is pd.DataFrame:
        data = data.values
      funnel_dict = self._pre({'array': data})
      tap_dict = ww.pour(funnel_dict, key_type='str')
      return tap_dict

    if data is not None and data_iter is None:
      return normalize_and_pour(data)
    elif data_iter is not None and data is None:
      return itertools.imap(normalize_and_pour, data_iter)

    else:
      raise ValueError("Must supply exactly one data or data_iter.")

  def tap_dict_to_examples(self, tap_dict, prefix=''):
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

    num_examples = tap_dict[tap_dict.keys()[0]].shape[0]

    # Get the dictionary of attributes (shape, dtype, etc.) of the arrays in
    # pour_outputs.
    att_dict = self._get_array_attributes(prefix)

    # Go through each row and each key of pour_outputs. Flatten the array and
    # convert it into it's proper feature. Return as list of dicts.

    example_dicts = []

    for row_num in xrange(num_examples):
      example_dict = {}

      for key in tap_dict:
        dtype = att_dict[key]['np_type']
        flat = tap_dict[key][row_num].flatten().astype(dtype)
        example_dict[key] = att_dict[key]['feature_func'](flat)

      example_dicts.append(example_dict)

    return example_dicts

  def tap_dict_from_pred(self, pred):
    raise NotImplementedError()

  def pump(self, tap_dict, df=False, index=None):
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
    funnel_dict = ww.pump(tap_dict, key_type='str')
    array = funnel_dict[self._pre('array')].astype(self.input_dtype)
    if df:
      if index is None:
        index = np.arange(array.shape[0])
      data = pd.DataFrame(
        data=array,
        index=index,
        columns=self.cols
      )
    else:
      data = array

    return data

  def examples_to_tap_dict(self, example_dicts, prefix=''):
    """Run the pump transformation on a list of example dictionaries to reconstruct the original array.

    Parameters
    ----------
    example_dicts: list of dicts of arrays or dict of arrays
      The example dictionaries which the arrays associated with a single example.

    Returns
    -------
    np.ndarray
      The numpy array to transform into examples.

    """

    if type(example_dicts) is not dict:
      arrays_dict = self._array_dicts_to_array_dict(example_dicts)
    else:
      arrays_dict = {}
      arrays_dict.update(example_dicts)
    att_dict = self._get_array_attributes(prefix)

    for key in arrays_dict:
      arrays_dict[key] = arrays_dict[key].reshape([-1] + att_dict[key]['shape'])
      arrays_dict[key] = arrays_dict[key].astype(att_dict[key]['np_type'])
    tap_dict = self._parse_examples(arrays_dict)

    return tap_dict

  def read_and_decode(self, serialized_example, prefix='', keep_features=None, drop_features=None):
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
    if drop_features is None:
      drop_features = []

    att_dict = self._get_array_attributes(prefix)
    feature_dict = {}
    shape_dict = {}

    if keep_features is not None:
      att_dict = {k: att_dict[k] for k in keep_features}

    for key in att_dict:
      if key in drop_features:
        continue

      feature_dict[key] = tf.FixedLenFeature(
        att_dict[key]['size'],
        att_dict[key]['tf_type']
      )
      shape_dict[key] = att_dict[key]['shape']

    features = tf.parse_single_example(
      serialized_example,
      features=feature_dict
    )

    for key in shape_dict:
      features[key] = tf.reshape(features[key], shape_dict[key])
      np_dtype = att_dict[key]['np_type']
      if np_dtype == np.float64:
        features[key] = tf.cast(features[key], dtype=tf.float64)

    return features

  def save_to_file(self, path):
    """Save the transform object to disk."""
    save_dict = self._save_dict()
    d.save_to_file(save_dict, path)

def is_lambda(obj):
  example_lambda = lambda a: a
  return isinstance(obj, type(example_lambda)) and obj.__name__ == example_lambda.__name__
