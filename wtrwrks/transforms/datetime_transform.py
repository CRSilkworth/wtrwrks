"""DateTimeTransform definition."""
import transform as n
import numpy as np
import datetime
import logging
import wtrwrks.tanks.tank_defs as td
import wtrwrks.read_write.tf_features as feat
from wtrwrks.waterworks.empty import empty
import tensorflow as tf


class DateTimeTransform(n.Transform):
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
  attribute_dict = {'norm_mode': None, 'norm_axis': None, 'num_units': 1, 'time_unit': 'D', 'fill_nat_func': lambda array: np.array(datetime.datetime(1970, 1, 1)), 'name': '', 'mean': None, 'std': None, 'min': None, 'max': None, 'dtype': np.float64, 'input_dtype': np.datetime64, 'zero_datetime': datetime.datetime(1970, 1, 1)}

  for k, v in n.Transform.attribute_dict.iteritems():
    if k in attribute_dict:
      continue
    attribute_dict[k] = v

  required_params = set([])
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
    super(DateTimeTransform, self).__init__(from_file, save_dict, **kwargs)

    # Ensure a valid norm mode was passed
    valid_norm_modes = ('mean_std', 'min_max', None)
    if self.norm_mode not in valid_norm_modes:
      raise ValueError("{} is an invalid norm_mode. Accepted norm mods are ".format(self.norm_mode, valid_norm_modes))

    # Ensure a valid norm_axis was passed
    valid_norm_axis = (0, 1, (0, 1), None)
    if self.norm_axis not in valid_norm_axis:
      raise ValueError("{} is an invalid norm_axis. Accepted norm axes are ".format(self.norm_axis, valid_norm_axis))

    # Ensure a valid time unit was supplied
    valid_time_units = ['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as']
    if self.time_unit not in valid_time_units:
      raise ValueError("{} is an invalid time_unit. Accepted time units are ".format(self.time_unit, valid_time_units))

    # Convert to np.datetime64
    if type(self.zero_datetime) is datetime.datetime:
      self.zero_datetime = np.datetime64(self.zero_datetime)

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
    if self.input_dtype is None:
      self.input_dtype = array.dtype
    else:
      array = np.array(array, dtype=self.input_dtype)
    array = array.astype(np.datetime64)

    batch_size = float(array.shape[0])
    total_examples = self.num_examples + batch_size

    # Get the inputted dtype
    if self.norm_mode == 'mean_std':
      # Find the means and standard deviations of each column
      array[np.isnat(array)] = self.fill_nat_func(array)
      temp_array = (array - self.zero_datetime)/np.timedelta64(self.num_units, self.time_unit)
      temp_array = temp_array.astype(self.dtype)

      if self.mean is None:
        self.mean = np.mean(temp_array, axis=self.norm_axis)
        self.var = np.var(temp_array, axis=self.norm_axis)
      else:
        self.mean = (self.num_examples / total_examples) * self.mean + (batch_size / total_examples) * np.mean(temp_array, axis=self.norm_axis)
        self.var = np.var(temp_array, axis=self.norm_axis)

    elif self.norm_mode == 'min_max':
      # Find the means and standard deviations of each column
      # temp_col_array = col_array[~np.isnat(col_array)]
      array[np.isnat(array)] = self.fill_nat_func(array)
      temp_array = (array - self.zero_datetime)/np.timedelta64(self.num_units, self.time_unit)
      temp_array = temp_array.astype(self.dtype)

      if self.min is None:
        self.min = np.min(temp_array, axis=self.norm_axis)
        self.max = np.max(temp_array, axis=self.norm_axis)
      else:
        self.min = np.minimum(self.min, np.min(temp_array, axis=self.norm_axis))
        self.max = np.maximum(self.max, np.max(temp_array, axis=self.norm_axis))

    self.num_examples += batch_size

  def _finish_calc(self):
    """Finish up the calc global value process."""
    if self.norm_mode == 'mean_std':
      self.std = np.sqrt(self.var)
      # If there are any standard deviations of 0, replace them with 1's,
      # print out a warning.
      if len(self.std[self.std == 0]):
        zero_stds = []
        for index in np.where(self.std == 0.0)[0]:
          zero_stds.append(index)

        logging.warn(self.name + " has zero-valued stds at " + str(zero_stds) + " replacing with 1's")

        self.std[self.std == 0] = 1.0
    elif self.norm_mode == 'min_max':
      # Test to make sure that min and max are not equal. If they are replace
      # with default values.
      if (self.min == self.max).any():
        if self.max.shape:
          self.max[self.max == self.min] = self.max[self.max == self.min] + 1
        else:
          self.max = self.max + 1

        logging.warn("DatetimeTransform " + self.name + " the same values for min and max, replacing with " + str(self.min) + " " + str(self.max) + " respectively.")

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
      'shape': list([len(self.cols)]),
      'tf_type': feat.select_tf_dtype(self.dtype),
      'size': feat.size_from_shape([len(self.cols)]),
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
    # Replace all the NaT's with the inputted replace_with.
    nats, nats_slots = td.isnat(array)
    nats_slots['a'].set_name('array')

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

    # Convert the datetimes to numbers
    nums, _ = td.datetime_to_num(replaced['target'], self.zero_datetime, self.num_units, self.time_unit, name='dtn')
    nums['diff'].set_name('diff')

    if self.norm_mode == 'mean_std':
      nums, _ = td.sub(
        nums['target'], self.mean,
        tube_plugs={'a_is_smaller': False, 'smaller_size_array': self.mean}
      )
      nums, _ = td.div(
        nums['target'], self.std,
        tube_plugs={'a_is_smaller': False, 'smaller_size_array': self.std, 'missing_vals': np.array([]), 'remainder': np.array([])}
      )
    elif self.norm_mode == 'min_max':
      nums, _ = td.sub(
        nums['target'], self.min,
        tube_plugs={'a_is_smaller': False, 'smaller_size_array': self.min}
      )
      nums, _ = td.div(
        nums['target'], (self.max - self.min),
        tube_plugs={'a_is_smaller': False, 'smaller_size_array': (self.max - self.min), 'missing_vals': np.array([]), 'remainder': np.array([])}
      )

    nums['target'].set_name('nums')

    if return_tubes is not None:
      ww = nums['target'].waterwork
      r_tubes = []
      for r_tube_key in return_tubes:
        r_tubes.append(ww.maybe_get_tube(r_tube_key))
      return r_tubes
