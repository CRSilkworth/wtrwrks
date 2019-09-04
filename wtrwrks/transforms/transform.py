"""Definition of the base Transform class"""
import pandas as pd
import wtrwrks.utils.dir_functions as d
import wtrwrks.utils.array_functions as af
import wtrwrks.waterworks.waterwork as wa
import os
import numpy as np
import tensorflow as tf
import itertools
import wtrwrks.utils.multiprocessing as mh
import wtrwrks.utils.batch_functions as b
import logging
import random
import glob


class Transform(object):
  """Abstract class used to create mappings from raw to vectorized, normalized data and vice versa. These transform store all the information necessary to create a Waterwork object which can do all the necessary reversible alterations to the data and also to transform it back to its original form.

  Parameters
  ----------
  name : str
    The name of the transform.
  dtype : numpy dtype
    The data type the transformed data should have. Defaults to np.float64.
  input_dtype: numpy dtype
    The datatype of the original inputted array.

  Attributes
  ----------
  attribute_dict: dict
    The keys are the attributes of the class while the values are the default values. It's done this way rather than defined in the __init__ because this dictionary also defines what values need to be saved when written to disk, and what values need to be displayed when printed to the terminal.
  required_params: set of strs
    The parameters that must be provided to the transform at definition.
  cols : list of strs
    The column names of the array or dataframe to be transformed.
  num_examples : int
    The number of rows that have passed through the calc_global_values function.
  is_calc_run : bool
    Whether or not calc_global_values has been run for this transform.

  """

  attribute_dict = {'name': '', 'cols': None, 'num_examples': None, 'is_calc_run': False, 'input_dtype': None, 'dtype': None}
  required_params = set(['name'])

  def __init__(self, from_file=None, save_dict=None, **kwargs):
    """Define a transform using a dictionary, file, or by setting the attribute values in kwargs.

    Parameters
    ----------
    from_file : None or str
      The file path of the Tranform that was written to disk.
    save_dict : dict or None
      The dictionary of attributes that completely define a Transform.
    name : str
      The name of the transform.
    dtype : numpy dtype
      The data type the transformed data should have. Defaults to np.float64.
    input_dtype: numpy dtype
      The datatype of the original inputted array.
    **kwargs :
      The keyword arguments that set the values of the attributes defined in the attribute_dict.

    """
    # Build from file
    if from_file is not None:
      save_dict = d.read_from_file(from_file)
      self._from_save_dict(save_dict)
    # Build from dict
    elif save_dict is not None:
      self._from_save_dict(save_dict)
    # Build from scratch
    else:
      # Check for invalid inputs
      attribute_set = set(self.attribute_dict)
      invalid_keys = sorted(set(kwargs.keys()) - attribute_set)
      if invalid_keys:
        raise TypeError("{} got unexpected keyword argument {}".format(self.__class__.__name__, invalid_keys[0]))

      # Set all the attributes passed to the constructor
      for key in self.attribute_dict:
        if key in kwargs:
          setattr(self, key, kwargs[key])
        else:
          setattr(self, key, self.attribute_dict[key])

      # Ensure all required parameters have been set
      for key in sorted(self.required_params):
        if key not in kwargs:
          raise TypeError("Must supply '{}' as an argument".format(key))

    self.waterwork = None

  def __str__(self):
    """Return the stringified values for each of the attributes in attribute list."""
    return str({a: str(getattr(self, a)) for a in self.attribute_dict})

  def __len__(self):
    raise NotImplementedError()

  def _finish_calc(self):
    """Finish up the calc global value process."""
    return

  def _from_save_dict(self, save_dict):
    """Reconstruct the transform object from the dictionary of attributes."""
    for key in self.attribute_dict:
      setattr(self, key, save_dict[key])

  def _get_array_attributes(self, prefix):
    raise NotImplementedError()

  def _get_dataset(self, file_name_pattern, batch_size, num_epochs=None, num_examples=None, filters=None, keep_features=None, drop_features=None, add_tensors=None, num_threads=1, shuffle_buffer_size=10000, random_seed=None):
    """Create the tensoflow dataset object to be used for input into training pipelines.

    Parameters
    ----------
    file_name_pattern : string
      The file name pattern of the tfrecord files. Supports '*', etc.
    batch_size : int
      The number of example to include in a single batch.
    num_epochs : int
      The number times to run through the full dataset of examples. Defaults to running indefinitely.
    num_examples : int
      Number of examples to run in the dataset before terminating. Cannot use when num_epochs is defined
    filters : dict of functions
      Any pre preprocessing filters to put on the dataset. The keys are the filter names, the values are the filter themselves
    keep_features : list of strs
      The features to keep after reading in from the tfrecords. Defaults to all of them.
    drop_features : list of strs
      The features to drop when reading from the tfrecords. Defaults to None. Can only use this or keep_features not both.
    add_tensors : dict of tensors
      Any additional tensors to add into the dataset object.
    num_threads : int
      The number of io threads to use. Defaults to 1, should probably not be more than 3.
    shuffle_buffer_size : int
      How many examples to shuffle together.
    random_seed : int or None
      The seed to set the random number generator

    Returns
    -------
    dataset : tf.dataset
      The dataset object to feed into tensorflow training pipelines.

    """
    # Set the random seed.
    random.seed(random_seed)

    # Get all the tfrecor file names and shuffle them
    file_names = glob.glob(file_name_pattern)
    file_names.sort()
    shuffled_file_names = random.sample(file_names, len(file_names))

    # Define the dataset object from the tfrecord files
    dataset = tf.data.TFRecordDataset(shuffled_file_names)

    # If num steps was given then use that to define how long to run the dataset
    if num_examples is not None:
      dataset = dataset.take(num_examples)
      dataset = dataset.shuffle(
          buffer_size=shuffle_buffer_size
      )
    # Otherwise use the number of epochs
    else:
      # Convert num epochs to integer
      if num_epochs is not None and type(num_epochs) is not int and not (tf.contrib.framework.is_tensor(num_epochs) and num_epochs.dtype is tf.int64):
        logging.warn(
          '%s is not a whole number. Will be converted to %s',
          num_epochs,
          int(num_epochs)
        )
        num_epochs = int(num_epochs)

      # Shuffle the dataset
      s_and_r = tf.data.experimental.shuffle_and_repeat(
          buffer_size=shuffle_buffer_size,
          count=num_epochs
      )
      dataset = dataset.apply(s_and_r)

    # Read and decode the dataset
    dataset = dataset.map(
        lambda se: self.read_and_decode(se, '', keep_features, drop_features),
        num_parallel_calls=num_threads
    )

    # If any filters were put in place filter the values.
    if filters is not None:
      for key in filters:
        dataset = dataset.filter(
          filters[key]
        )

    # Batch out the data
    if batch_size is not None:
      dataset = dataset.batch(batch_size)

    # Add any additional tensors to the dataset.
    if add_tensors is not None:
      for key in add_tensors:
        def _add_tensor(kwargs):
          kwargs[key] = add_tensors[key]
          return kwargs
        dataset = dataset.map(_add_tensor)
    return dataset

  def _maybe_convert_to_df(self, data):
    """Convert to pandas dataframe if data is array, otherwise do nothing."""
    if type(data) is pd.DataFrame:
      return data
    else:
      return pd.DataFrame(
        data=data,
        columns=self.cols,
        index=np.arange(data.shape[0])
      )

  def _nopre(self, to_unprefix, prefix=''):
    """Strip the self.name/prefix from a string or keys of a dictionary.

    Parameters
    ----------
    to_unprefix : str or dict
      The string or dictionary to strip the prefix/self.name  from.
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    unprefixed : str or dict
      The string with the prefix/self.name stripped or the dictionary with the prefix/self.name stripped from all the keys.

    """
    str_len = len(os.path.join(prefix, self.name) + '/')
    if str_len == 1:
      str_len = 0
    if type(to_unprefix) is not dict:
      return to_unprefix[str_len:]

    r_d = {}
    for key in to_unprefix:
      if type(key) is tuple and type(key[0]) in (str, unicode):
        r_d[(key[0][str_len:], key[1])] = to_unprefix[key]
      elif type(key) in (str, unicode):
        r_d[key[str_len:]] = to_unprefix[key]
      else:
        r_d[key] = to_unprefix[key]
    return r_d

  def _pre(self, to_prefix, prefix=''):
    """Add the name and some additional prefix to the keys in a dictionary or to a string directly.

    Parameters
    ----------
    to_prefix : str or dict
      The string or dictionary to add to the prefix/self.name prefix to.
    prefix : str
      Any additional prefix to give the string/dictionary keys. Defaults to no additional prefix.

    Returns
    -------
    str or dict
      The string with the prefix/self.name added or the dictionary with the prefix/self.name added to all the keys.

    """
    if type(to_prefix) is not dict:
      return os.path.join(prefix, self.name, to_prefix)

    r_d = {}
    for key in to_prefix:
      if type(key) is tuple and type(key[0]) in (str, unicode):
        r_d[(os.path.join(prefix, self.name, key[0]), key[1])] = to_prefix[key]
      elif type(key) in (str, unicode):
        r_d[os.path.join(prefix, self.name, key)] = to_prefix[key]
      else:
        r_d[key] = to_prefix[key]
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
    """Start the calc global value process."""
    self.num_examples = 0.

  def calc_global_values(self, data=None, data_iter=None):
    """Calculate any values needed for the transform that require information from the entired dataset.

    Parameters
    ----------
    data : np.array or pd.DataFrame
      The entire dataset in the form of a numpy array or a pandas DataFrame. Should have the same columns as the arrays that will be fed to the pour method.
    data_iter : iterator of np.array or pd.DataFrame
      The entire dataset in the form of an iterator of numpy array or a pandas DataFrame. Needed if the dataset is too large to fit in memory. Should have the same columns as the arrays that will be fed to the pour method. Can only use if 'data' is not being used

    """
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

  def define_waterwork(self, array=None, return_tubes=None, prefix=''):
    """Get the waterwork that completely describes the pour and pump transformations.

    Parameters
    ----------
    array : np.ndarray or empty
      The array to be transformed.
    return_tubes : list of str or None
      Tube objects to be returned from the Waterwork object. Only needed if Waterworks are being stiched together.
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    Waterwork
      The waterwork with all the tanks (operations) added, and names set.

    """
    raise NotImplementedError()

  def examples_to_tap_dict(self, example_dicts, prefix=''):
    """Run the pump transformation on a list of example dictionaries to reconstruct the original array.

    Parameters
    ----------
    example_dicts: list of dicts of arrays or dict of arrays
      The example dictionaries which the arrays associated with a single example.

    Returns
    -------
    tap_dict: dict
      The dictionary all information needed to completely reconstruct the original data.

    """
    arrays_dict = {}

    # If the examples are all broken up into individual rows then put them into
    # one large dictionary of arrays
    if type(example_dicts) is not dict:
      for array_dict in example_dicts:
        for key in array_dict:
          arrays_dict.setdefault(key, [])
          arrays_dict[key].append(array_dict[key])

      for key in arrays_dict:
        arrays_dict[key] = np.stack(arrays_dict[key])
    else:
      arrays_dict.update(example_dicts)

    # Do any necessary reshaping and recasting
    att_dict = self._get_array_attributes(prefix)
    for key in arrays_dict:
      arrays_dict[key] = arrays_dict[key].reshape([-1] + att_dict[key]['shape'])
      arrays_dict[key] = arrays_dict[key].astype(att_dict[key]['np_type'])

    return arrays_dict

  def get_default_array(self, batch_size=1):
    """Get an array of the proper shape and type that matches the input array.

    Parameters
    ----------
    batch_size : int
      The number of example to include in a single batch.

    Returns
    -------
    default_array : np.array
      The numpy array all filled with some default value

    """
    #####################################
    # NOTE: PUT BACK IN CHECK AFTER WORKFLOW IS RERUN
    # assert self.is_calc_run, ("Run calc_global_values before getting default array")
    ####################################
    return af.empty_array([batch_size, len(self.cols)], self.input_dtype)

  def get_dataset_iter(self, file_name_pattern, batch_size, keep_features=None, drop_features=None, add_tensors=None):
    """Create the tensoflow dataset iterator object used iterator over a tensorflow dataset object.

    Parameters
    ----------
    file_name_pattern : string
      The file name pattern of the tfrecord files. Supports '*', etc.
    batch_size : int
      The number of example to include in a single batch.
    add_tensors : dict of tensors
      Any additional tensors to add into the dataset object.

    Returns
    -------
    dataset_iter : tf.data.Iterator
      The dataset object to feed into tensorflow training pipelines.

    """
    dataset = self._get_dataset(file_name_pattern, batch_size, keep_features=keep_features, drop_features=drop_features, add_tensors=add_tensors)

    data_iter = tf.data.Iterator.from_structure(
        dataset.output_types,
        dataset.output_shapes
    )

    return data_iter

  def get_dataset_feed_iter(self, file_name_pattern, batch_size, keep_features=None, drop_features=None, add_tensors=None):
    """Create the tensoflow dataset iterator object used iterator over a tensorflow dataset object. Used in input pipelines where a feed dict is used.

    Parameters
    ----------
    file_name_pattern : string
      The file name pattern of the tfrecord files. Supports '*', etc.
    batch_size : int
      The number of example to include in a single batch.
    add_tensors : dict of tensors
      Any additional tensors to add into the dataset object.

    Returns
    -------
    dataset_iter : tf.data.Iterator
      The dataset object to feed into tensorflow training pipelines.

    """
    dataset = self._get_dataset(file_name_pattern, batch_size, keep_features=keep_features, drop_features=drop_features, add_tensors=add_tensors)
    handle = tf.placeholder(tf.string, shape=[])

    feed_iter = tf.data.Iterator.from_string_handle(
        handle,
        dataset.output_types,
        dataset.output_shapes
    )

    return feed_iter, handle

  def get_dataset_iter_init(self, dataset_iter, file_name_pattern, batch_size, num_epochs=None, num_examples=None, filters=None, keep_features=None, drop_features=None, add_tensors=None, num_threads=1, shuffle_buffer_size=1000, random_seed=None):
    """Create the tensoflow dataset object to be used for input into training pipelines.

    Parameters
    ----------
    dataset_iter : tf.data.Iterator
      The dataset object to feed into tensorflow training pipelines.
    file_name_pattern : string
      The file name pattern of the tfrecord files. Supports '*', etc.
    batch_size : int
      The number of example to include in a single batch.
    num_epochs : int
      The number times to run through the full dataset of examples. Defaults to running indefinitely.
    num_examples : int
      Number of steps to run in the dataset before terminating. Cannot use when num_epochs is defined
    filters : dict of functions
      Any pre preprocessing filters to put on the dataset. The keys are the filter names, the values are the filter themselves
    keep_features : list of strs
      The features to keep after reading in from the tfrecords. Defaults to all of them.
    drop_features : list of strs
      The features to drop when reading from the tfrecords. Defaults to None. Can only use this or keep_features not both.
    add_tensors : dict of tensors
      Any additional tensors to add into the dataset object.
    num_threads : int
      The number of io threads to use. Defaults to 1, should probably not be more than 3.
    shuffle_buffer_size : int
      How many examples to shuffle together.
    random_seed : int or None
      The seed to set the random number generator

    Returns
    -------
    dataset : tf.dataset
      The dataset object to feed into tensorflow training pipelines.

    """
    dataset = self._get_dataset(file_name_pattern, batch_size, num_epochs, num_examples, filters, keep_features, drop_features, add_tensors, num_threads, shuffle_buffer_size, random_seed)

    return dataset_iter.make_initializer(dataset)

  def get_placeholder(self, tap_key, with_batch=True, batch_size=None):
    """Get a placehold associated with some tap of this transform.

    Parameters
    ----------
    tap_key : str
      The name of the tap to get a placeholder of.
    with_batch : bool
      Whether or not to add a batch dimension
    batch_size : int
      The size of a batch. Defaults to None (i.e. variable).

    Returns
    -------
    ph : tf.placeholder
      The created placeholder object.

    """
    att_dict = self._get_array_attributes()

    # Create the proper shape
    if with_batch:
      shape = [batch_size] + att_dict[tap_key]['shape']
    else:
      shape = att_dict[tap_key]['shape']

    # Get the proper dtype
    if att_dict[tap_key]['np_type'] == np.float64:
      dtype = tf.float64
    else:
      dtype = att_dict[tap_key]['tf_type']

    ph = tf.compat.v1.placeholder(
      dtype=dtype,
      shape=shape
    )

    return ph

  def get_schema_dict(self, var_lim=255):
    """Create a dictionary which defines the proper fields which are needed to store the untransformed data in a (postgres) SQL database.

    Parameters
    ----------
    var_lim : int or dict
      The maximum size of strings. If an int is passed then all VARCHAR fields have the same limit. If a dict is passed then each field gets it's own limit. Defaults to 255 for all string fields.

    Returns
    -------
    schema_dict : dict
      Dictionary where the keys are the field names and the values are the SQL data types.

    """
    # Select the SQL datatype by the input_dtype
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

    # Create the schema dictionary.
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

  def pour(self, data=None, data_iter=None):
    """Execute the transformation in the pour (forward) direction.

    Parameters
    ----------
    data : np.ndarray
      The numpy array to transform.
    data_iter : iterator of np.array or pd.DataFrame
      The entire dataset in the form of an iterator of numpy array or a pandas DataFrame. Needed if the dataset is too large to fit in memory. Should have the same columns as the arrays that will be fed to the pour method. Can only use if 'data' is not being used
    Returns
    -------
    tap_dict : dict (or iterator of dicts)
      The dictionary of transformed outputs as well as any additional information needed to completely reconstruct the original data. Returns an iterator of dicts if something is passed to 'data_iter' rather than the 'data' argument.

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

  def pump(self, tap_dict, df=False, index=None):
    """Execute the transformation in the pump (backward) direction.

    Parameters
    ----------
    tap_dict: dict
      The dictionary all information needed to completely reconstruct the original data.

    Returns
    -------
    data : np.ndarray or df
      The original numpy array or pandas dataframe that was poured.

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

  def read_and_decode(self, serialized_example, prefix='', keep_features=None, drop_features=None):
    """Convert a serialized example created from an example dictionary from this transform into a dictionary of shaped tensors for a tensorflow pipeline.

    Parameters
    ----------
    serialized_example : tfrecord serialized example
      The serialized example to read and convert to a dictionary of tensors.
    prefix : str
      A string to prefix the dictionary keys with.
    keep_features : list of strs
      The features to keep after reading in from the tfrecords. Defaults to all of them.
    drop_features : list of strs
      The features to drop when reading from the tfrecords. Defaults to None. Can only use this or keep_features not both.

    Returns
    -------
    features : dict of tensors
      The tensors created by decoding the serialized example

    """
    feature_dict = {}
    shape_dict = {}

    att_dict = self._get_array_attributes(prefix)

    # Define the set of features that will be kept
    if drop_features is None:
      drop_features = []
    if keep_features is not None:
      att_dict = {k: att_dict[k] for k in keep_features}

    for key in att_dict:
      if key in drop_features:
        continue

      # Define the tf feature object and pull out the correct shape
      feature_dict[key] = tf.FixedLenFeature(
        att_dict[key]['size'],
        att_dict[key]['tf_type']
      )
      shape_dict[key] = att_dict[key]['shape']

    # Parse the example
    features = tf.parse_single_example(
      serialized_example,
      features=feature_dict
    )

    # Do any reshaping and cast to float if the numpy dtype is np.float64
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

  def tap_dict_to_examples(self, tap_dict, prefix=''):
    """Run the pour transformation on an array to transform it into a form best for ML pipelines. This list of example dictionaries can be easily converted into tf records, but also have all the information needed in order to reconstruct the original array.

    Parameters
    ----------
    tap_dict : dict (or iterator of dicts)
      The dictionary of transformed outputs as well as any additional information needed to completely reconstruct the original data. Returns an iterator of dicts if something is passed to 'data_iter' rather than the 'data' argument.

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

  def write_examples(self, data=None, data_iter=None, file_name=None, file_num_offset=0, batch_size=1, num_threads=1, skip_fails=False, skip_keys=None, use_threading=False, serialize_func=None, prefix=''):
    """Pours the arrays then writes the examples to tfrecords in a multithreading manner. It creates one example per 'row', i.e. axis=0 of the arrays. All arrays must have the same axis=0 dimension and must be of a type that can be written to a tfrecord

    Parameters
    ----------
    data : np.array or pd.DataFrame
      The entire dataset in the form of a numpy array or a pandas DataFrame. Should have the same columns as the arrays that will be fed to the pour method.
    data_iter : iterator of np.array or pd.DataFrame
      The entire dataset in the form of an iterator of numpy array or a pandas DataFrame. Needed if the dataset is too large to fit in memory. Should have the same columns as the arrays that will be fed to the pour method. Can only use if 'data' is not being used
    file_name : str
      The name of the tfrecord file to write to. An extra '_<num>' will be added to the name.
    file_num_offset : int
      A number that controls what number will be appended to the file name (so that files aren't overwritten.)
    batch_size : int
      The number of example to include in a single batch.
    num_threads : int
      The number of io threads to use. Defaults to 1, should probably not be more than 3.
    skip_fails : bool
      Whether or not to skip any write failures without error. Defaults to false.
    skip_keys : list of strs
      Any taps that should not be written to examples.
    use_threading : bool
      Whether or not to use multithreading rather than multiprocessing. Defaults to False
    serialize_func : np.array -> list of serialized examples
      The function that handles the transformation from numpy array to serialized example, if the user does not want to use the default one.
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.
    """
    # Make sure only data or data_iter is passed
    if data is not None and data_iter is None:
      data_iter = [data]
    elif data_iter is not None and data is None:
      pass
    else:
      raise ValueError("Must supply exactly one data or data_iter.")

    # If no serialize_func is passed then define the default one.
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

    # If the data iterator is a list, then convert to tuple
    if type(data_iter) in (list, tuple):
      data_iter = (i for i in data_iter)

    # Make sure the file name ends in tfrecord
    if not file_name.endswith('.tfrecord'):
      raise ValueError("file_name must end in '.tfrecord'")

    # Create the directory if it doesn't exist
    dir = '/'.join(file_name.split('/')[:-1])
    if dir:
      d.maybe_create_dir(dir)

    # Batch out the data iterator into batches of batch size
    file_names = []
    for batch_num, batch in enumerate(b.batcher(data_iter, batch_size)):
      logging.info("Serializing batch %s", batch_num)

      # If skip fails then catch any exceptions otherwise just run the batch
      # through the serialize function
      if skip_fails:
        try:
          all_serials = mh.multi_map(serialize_func, batch, num_threads, use_threading)
        except Exception:
          logging.warn("Batched %s failed. Skipping.", batch_num)
          continue
      else:
        all_serials = mh.multi_map(serialize_func, batch, num_threads, use_threading)

      logging.info("Finished serializing batch %s", batch_num)

      # Add to the full list of created files.
      file_num = file_num_offset + batch_num
      fn = file_name.replace('.tfrecord', '_' + str(file_num) + '.tfrecord')
      file_names.append(fn)

      # Write the examples to disk
      logging.info("Writing batch %s", batch_num)
      writer = tf.io.TFRecordWriter(fn)
      for serials in all_serials:
        for serial in serials:
          writer.write(serial)

      # Close the writer
      logging.info("Finished writing batch %s", batch_num)
      writer.close()

    return file_names
