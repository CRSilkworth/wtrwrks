"""FourierTransform definition."""
import transform as n
import numpy as np
import datetime
import logging
import wtrwrks.tanks.tank_defs as td
import wtrwrks.read_write.tf_features as feat
from wtrwrks.waterworks.empty import empty
import tensorflow as tf
import sqlalchemy as sa


class FourierTransform(n.Transform):
  """Class used to create mappings from raw datetime data to vectorized, normalized data and vice versa.

  Parameters
  ----------
  name : str
    The name of the transform.
  dtype : numpy dtype
    The data type the transformed data should have. Defaults to np.float64.
  input_dtype: numpy dtype
    The datatype of the original inputted array.
  norm_mode : "mean_std", "min_max" or None
    How to normalize the data. Subtracting out the mean and dividing by the standard deviation; Subtracting out min and dividing by the difference between max and min; or leaving as is.
  norm_axis : 0, 1, or None
    In the case that 'norm_mode' has a non None value, what axis should be used for the normalization. None implies both 0 and 1.
  time_unit : str in ['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as']
    The time resolution to break up the datetime values into. They are numpy tiem units.
  fill_nat_func : np.array(dtype=datetime.datetime) -> np.array(dtype=datetime.datetime)
    Function which fills in NaT values with some value. This function takes in a numpy array which contains NaT and returns one whith those values replaced with valid values.
  zero_datetime : datetime.datetime
    The time that will be defined as 0 when transformed. defaults to 1970-1-1.

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
  mean : numpy array
    The stored mean values to be used to normalize data.
  std : numpy array
    The stored standard deviation values to be used to normalize data.
  min : numpy array
    The stored mins to be used to normalize data.
  max : numpy array
    The stored maxes to be used to normalize data.

  """
  attribute_dict = {'num_units': 1, 'time_unit': 'D', 'fill_nat_func': lambda array: np.array(datetime.datetime(1970, 1, 1)), 'fill_nan_func': lambda array: np.array(0.0), 'name': '', 'dtype': np.float64, 'input_dtype': np.object, 'zero_datetime': None, 'end_datetime': None, 'num_frequencies': None, 'top_frequencies': None, 'X_k': None, 'w_k': None}

  for k, v in n.Transform.attribute_dict.iteritems():
    if k in attribute_dict:
      continue
    attribute_dict[k] = v

  required_params = set(['zero_datetime', 'end_datetime', 'num_frequencies'])
  required_params.update(n.Transform.required_params)

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
    norm_mode : "mean_std", "min_max" or None
      How to normalize the data. Subtracting out the mean and dividing by the standard deviation; Subtracting out min and dividing by the difference between max and min; or leaving as is.
    norm_axis : 0, 1, or None
      In the case that 'norm_mode' has a non None value, what axis should be used for the normalization. None implies both 0 and 1.
    time_unit : str in ['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as']
      The time resolution to break up the datetime values into. They are numpy tiem units.
    fill_nat_func : np.array(dtype=datetime.datetime) -> np.array(dtype=datetime.datetime)
      Function which fills in NaT values with some value. This function takes in a numpy array which contains NaT and returns one whith those values replaced with valid values.
    zero_datetime : datetime.datetime
      The time that will be defined as 0 when transformed. defaults to 1970-1-1.
    **kwargs :
      The keyword arguments that set the values of the attributes defined in the attribute_dict.

    """
    super(FourierTransform, self).__init__(from_file, save_dict, **kwargs)

    # Ensure a valid time unit was supplied
    valid_time_units = ['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as']
    if self.time_unit not in valid_time_units:
      raise ValueError("{} is an invalid time_unit. Accepted time units are ".format(self.time_unit, valid_time_units))

    if self.top_frequencies is None:
      self.top_frequencies = self.num_frequencies

    if self.top_frequencies > self.num_frequencies:
      raise ValueError("top_frequencies must be less than or equal to num_frequencies")

    # Convert to np.datetime64
    if type(self.zero_datetime) is np.ndarray and self.zero_datetime.dtype is not np.datetime64:
      self.zero_datetime = self.zero_datetime.astype(np.datetime64)
    elif type(self.zero_datetime) is datetime.datetime:
      self.zero_datetime = np.datetime64(self.zero_datetime)
    else:
      self.zero_datetime = np.array(self.zero_datetime, dtype=np.datetime64)

    if type(self.end_datetime) is np.ndarray and self.end_datetime.dtype is not np.datetime64:
      self.end_datetime = self.end_datetime.astype(np.datetime64)
    elif type(self.end_datetime) is datetime.datetime:
      self.end_datetime = np.datetime64(self.end_datetime)
    else:
      self.end_datetime = np.array(self.end_datetime, dtype=np.datetime64)

  def __len__(self):
    """Get the length of the vector outputted by the row_to_vector method."""
    # assert self.is_calc_run, ("Must run calc_global_values before taking the len.")
    return len(self.cols)

  def _calc_global_values(self, array):
    """Calculate all the values of the Transform that are dependent on all the examples of the dataset. (e.g. mean, standard deviation, unique category values, etc.) This method must be run before any actual transformation can be done.

    Parameters
    ----------
    array : np.ndarray
      Some of the data that will be transformed.

    """
    self.input_dtype = array.dtype

    if len(array.shape) < 2 or array.shape[1] != 2:
      raise ValueError("Array must have exactly two columns. The first being the time, and the second being the amplitude.")

    batch_size = float(array.shape[0])
    total_examples = self.num_examples + batch_size

    if not batch_size:
      return

    time_array = array[:, 0].astype(np.datetime64)
    time_array[np.isnat(time_array)] = self.fill_nat_func(time_array)

    # Get the maximum time and convert it to dtype
    end_datetime = (self.end_datetime - self.zero_datetime) / np.timedelta64(self.num_units, self.time_unit)
    end_datetime = end_datetime.astype(self.dtype)

    # Convert to dtype and scale to values between 0 and 1
    time_array = (time_array - self.zero_datetime)/np.timedelta64(self.num_units, self.time_unit)
    time_array = time_array.astype(self.dtype)
    time_array = time_array / end_datetime

    amp_array = array[:, 1: 2].astype(np.float64)
    amp_array[np.isnan(amp_array)] = self.fill_nan_func(amp_array)
    amp_array = np.tile(amp_array, [1, self.num_frequencies])

    exp = np.exp(-2.0j * np.pi * np.tensordot(time_array, self.w_k, [[], []]))

    self.X_k = batch_size / total_examples * np.mean(amp_array * exp, axis=0) + self.num_examples / total_examples * self.X_k

    self.num_examples += batch_size

  def _finish_calc(self):
    """Finish up the calc global value process."""

    sort_inds = np.argsort(np.abs(self.X_k))[::-1]
    self.X_k = self.X_k[sort_inds]
    self.w_k = self.w_k[sort_inds]

  def _get_array_attributes(self, prefix=''):
    """Get the dictionary that contain the original shapes of the arrays before being converted into tfrecord examples.

    Parameters
    ----------
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    array_attributes : dict
      The dictionary with keys equal to those that are found in the Transform's example dicts and values are the shapes of the arrays of a single example.

    """
    att_dict = {}
    att_dict['nums'] = {
      'shape': list([self.top_frequencies]),
      'tf_type': feat.select_tf_dtype(self.dtype),
      'size': feat.size_from_shape([self.top_frequencies]),
      'feature_func': feat.select_feature_func(self.dtype),
      'np_type': self.dtype
    }
    att_dict['amps'] = {
      'shape': [],
      'tf_type': feat.select_tf_dtype(self.dtype),
      'size': feat.size_from_shape([]),
      'feature_func': feat.select_feature_func(self.dtype),
      'np_type': self.dtype
    }
    att_dict['div'] = {
      'shape': list([self.top_frequencies]),
      'tf_type': feat.select_tf_dtype(self.dtype),
      'size': feat.size_from_shape([self.top_frequencies]),
      'feature_func': feat.select_feature_func(self.dtype),
      'np_type': self.dtype
    }
    att_dict['nats'] = {
      'shape': list([len(self.cols)]),
      'tf_type': tf.int64,
      'size': feat.size_from_shape([len(self.cols)]),
      'feature_func': feat._int_feat,
      'np_type': np.bool
    }
    att_dict['diff'] = {
      'shape': list([len(self.cols)]),
      'tf_type': tf.int64,
      'size': feat.size_from_shape([len(self.cols)]),
      'feature_func': feat._int_feat,
      'np_type': np.int64
    }

    att_dict = self._pre(att_dict, prefix)
    return att_dict

  def _get_eval_cls_cols(self, already_added_cols=None):
    """Get a dictionary of the sqlalchemy column types"""

    if already_added_cols is None:
      already_added_cols = set()

    eval_cls_cols = {}
    if self.cols[0] not in already_added_cols:
      eval_cls_cols[self.cols[0]] = sa.Column(sa.DateTime)
    if self.cols[1] not in already_added_cols:
      eval_cls_cols[self.cols[1]] = sa.Column(sa.Float)

    already_added_cols.add(self.cols[0])
    already_added_cols.add(self.cols[1])

    return eval_cls_cols

  def _start_calc(self):
    """Start the calc global value process."""
    # Create the mapping from category values to index in the vector and
    # vice versa

    self.w_k = np.arange(self.num_frequencies, dtype=np.float64)
    self.X_k = np.zeros([self.num_frequencies], dtype=np.complex)
    self.num_examples = 0.

  def define_waterwork(self, array=empty, return_tubes=None, prefix=''):
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
    splits_slots['a'].set_name('array')

    splits, _ = td.iter_list(splits['target'], 2)
    splits[1].set_name('amps')

    times, _ = td.reshape(
      splits[0],
      slot_plugs={'shape': lambda r: r[self._pre('array', prefix)].shape[:1]},
      tube_plugs={'old_shape': lambda r: list(r[self._pre('nums', prefix)].shape[:1]) + [1]}
    )
    times, _ = td.cast(
      times['target'], np.datetime64,
      tube_plugs={
        'input_dtype': self.input_dtype,
        'diff': np.array([], dtype=self.input_dtype)
      }
    )
    # Replace all the NaT's with the inputted replace_with.
    nats, nats_slots = td.isnat(times['target'])

    replaced, _ = td.replace(
      nats['a'], nats['target'],
      slot_plugs={
        'replace_with': lambda z: self.fill_nat_func(z[self._pre('array', prefix)])
      },
      tube_plugs={
        'replace_with': np.array([]),
        'replaced_vals': np.array([None], dtype=np.datetime64)
      }
    )

    replaced['replaced_vals'].set_name('replaced_vals')
    replaced['mask'].set_name('nats')

    end = (self.end_datetime - self.zero_datetime) / np.timedelta64(self.num_units, self.time_unit)
    end = end.astype(self.dtype)

    # Convert the datetimes to numbers
    nums, _ = td.datetime_to_num(replaced['target'], self.zero_datetime, self.num_units, self.time_unit, name='dtn')

    nums['diff'].set_name('diff')

    # nums, _ = td.sub(
    #   nums['target'], 0.0,
    #   tube_plugs={'a_is_smaller': False, 'smaller_size_array': 0.0}
    # )
    nums, _ = td.div(
      nums['target'], end,
      tube_plugs={'a_is_smaller': False, 'smaller_size_array': end, 'missing_vals': np.array([]), 'remainder': np.array([])}
    )

    decomp, _ = td.phase_decomp(
      nums['target'], self.w_k[:self.top_frequencies],
    )

    decomp['div'].set_name('div')
    decomp['target'].set_name('nums')

    if return_tubes is not None:
      ww = decomp['target'].waterwork
      r_tubes = []
      for r_tube_key in return_tubes:
        r_tubes.append(ww.maybe_get_tube(r_tube_key))
      return r_tubes

  def get_schema_dict(self, var_lim=None):
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
    schema_dict = {
      self.cols[0]: 'TIMESTAMP',
      self.cols[1]: 'FLOAT'
    }
    return schema_dict
