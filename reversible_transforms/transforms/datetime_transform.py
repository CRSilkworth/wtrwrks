import transform as n
import pandas as pd
import numpy as np
import datetime
import warnings
import reversible_transforms.tanks.tank_defs as td
import reversible_transforms.read_write.tf_features as feat
from reversible_transforms.waterworks.empty import empty
import os
import numpy as np
import tensorflow as tf
import pprint

class DateTimeTransform(n.Transform):
  """Class used to create mappings from raw datetime data to vectorized, normalized data and vice versa.

  Parameters
  ----------
  df : pd.DataFrame
    The dataframe with all the data used to define the mappings.
  columns : list of strs
    The column names of all the relevant columns that make up the data to be taken from the dataframe
  start_datetimes: list of datetimes
    The 'zero times' for each of the columns. Must be the same length as columns
  from_file : str
    The path to the saved file to recreate the transform object that was saved to disk.
  save_dict : dict
    The dictionary to rereate the transform object

  Attributes
  ----------
  attribute_list : list of strs
    The list of attributes that need to be saved in order to fully reconstruct the transform object.

  """
  attribute_dict = {'norm_mode': None, 'norm_axis': None, 'num_units': 1, 'time_unit': 'D', 'fill_nat_func': None, 'name': '', 'mean': None, 'std': None, 'min': None, 'max': None, 'dtype': np.float64, 'input_dtype': None, 'zero_datetime': datetime.datetime(1970, 1, 1)}

  def _setattributes(self, **kwargs):
    super(DateTimeTransform, self)._setattributes(**kwargs)

    if self.norm_mode not in (None, 'min_max', 'mean_std'):
      raise ValueError(self.norm_mode + " not a valid norm mode.")

    if type(self.zero_datetime) is datetime.datetime:
      self.zero_datetime = np.datetime64(self.zero_datetime)

    # if self.fill_nat_func is None:
    #   self.fill_nat_func = lambda array: np.full(array[np.isnat(array)].shape, self.zero_datetime)

  def calc_global_values(self, array, verbose=True):
    """Set all the relevant attributes for this subclass. Called by the constructor for the Transform class.

    Parameters
    ----------
    df : pd.DataFrame
      The dataframe with all the data used to define the mappings.
    columns : list of strs
      The column names of all the relevant columns that make up the data to be taken from the dataframe
    start_datetimes: list of datetimes
      The 'zero times' for each of the columns. Must be the same length as columns

    """
    array = array.astype(np.datetime64)
    # Get the inputted dtype
    self.input_dtype = array.dtype
    self.input_shape = array.shape

    if self.norm_mode == 'mean_std':
      # Find the means and standard deviations of each column
      temp_array = (array - self.zero_datetime)/np.timedelta64(self.num_units, self.time_unit)
      temp_array = temp_array.astype(self.dtype)

      if not len(temp_array):
        raise ValueError("Inputted col_array has no non nan values.")

      self.mean = np.nanmean(temp_array, axis=self.norm_axis)
      self.std = np.nanstd(temp_array, axis=self.norm_axis)

      # If any of the standard deviations are 0, replace them with 1's and
      # print out a warning
      if (self.std == 0).any():
        if verbose:
          warnings.warn("DatetimeTransform " + self.name + " has a zero-valued std, replacing with 1.")
        self.std[self.std == 0] = 1.0

    elif self.norm_mode == 'min_max':
      # Find the means and standard deviations of each column
      # temp_col_array = col_array[~np.isnat(col_array)]
      temp_array = (array - self.zero_datetime)/np.timedelta64(self.num_units, self.time_unit)
      temp_array = temp_array.astype(self.dtype)

      if not len(temp_array):
        raise ValueError("Inputted col_array has no non nan values.")

      self.min = np.nanmin(temp_array, axis=self.norm_axis)
      self.max = np.nanmax(temp_array, axis=self.norm_axis)

      # Test to make sure that min and max are not equal. If they are replace
      # with default values.
      if (self.min == self.max).any():
        self.max[self.max == self.min] = self.max[self.max == self.min] + 1

        if verbose:
          warnings.warn("DatetimeTransform " + self.name + " the same values for min and max, replacing with " + str(self.min) + " " + str(self.max) + " respectively.")

  def define_waterwork(self, array=empty):
    # Replace all the NaT's with the inputted replace_with.
    nats, _ = td.isnat(array)

    replaced, _ = td.replace(nats['a'], nats['target'])

    replaced['replaced_vals'].set_name('replaced_vals')
    replaced['mask'].set_name('nats')

    nums, _ = td.datetime_to_num(replaced['target'], self.zero_datetime, self.num_units, self.time_unit, name='dtn')
    nums['diff'].set_name('diff')

    if self.norm_mode == 'mean_std':
      nums, _ = nums['target'] - self.mean
      nums, _ = nums['target'] / self.std
    elif self.norm_mode == 'min_max':
      nums, _ = nums['target'] - self.min
      nums, _ = nums['target'] / (self.max - self.min)

    nums['target'].set_name('nums')

  def _get_funnel_dict(self, array=None, prefix=''):
    if self.fill_nat_func is None:
      fill_nat_func = lambda a: np.full(a[np.isnat(a)].shape, self.zero_datetime)
    else:
      fill_nat_func = self.fill_nat_func

    funnel_dict = {
      'Replace_0/slots/replace_with': fill_nat_func(array)
    }
    if array is not None:
      funnel_dict['IsNat_0/slots/a'] = array

    return self._pre(funnel_dict, prefix)

  def _extract_pour_outputs(self, tap_dict, prefix=''):
    return {self._pre(k, prefix): tap_dict[self._pre(k, prefix)] for k in ['nums', 'nats', 'diff']}

  def _get_tap_dict(self, pour_outputs, prefix=''):
    pour_outputs = self._nopre(pour_outputs, prefix)
    num_nats = len(np.where(pour_outputs['nats'])[0])
    tap_dict = {
      'nums': pour_outputs['nums'],
      'nats': pour_outputs['nats'],
      'replaced_vals': np.full([num_nats], 'NaT', dtype=self.input_dtype),
      'diff': pour_outputs['diff'],
      'dtn/tubes/zero_datetime': self.zero_datetime,
      'dtn/tubes/time_unit': self.time_unit,
      'dtn/tubes/num_units': self.num_units,
      'Replace_0/tubes/replace_with_shape': (num_nats,),
    }
    if self.norm_mode == 'mean_std' or self.norm_mode == 'min_max':
      if self.norm_mode == 'mean_std':
        sub_val = self.mean
        div_val = self.std
      else:
        sub_val = self.min
        div_val = self.max - self.min
      norm_mode_dict = {
        'Sub_0/tubes/smaller_size_array': sub_val,
        'Sub_0/tubes/a_is_smaller': False,
        'Div_0/tubes/smaller_size_array': div_val,
        'Div_0/tubes/a_is_smaller': False,
        'Div_0/tubes/remainder': np.array([], dtype=self.input_dtype),
        'Div_0/tubes/missing_vals': np.array([], dtype=float)
      }
      tap_dict.update(norm_mode_dict)

    return self._pre(tap_dict, prefix)

  def _extract_pump_outputs(self, funnel_dict, prefix=''):
    return funnel_dict[self._pre('IsNat_0/slots/a', prefix)]

  def _get_example_dicts(self, pour_outputs, prefix=''):
    pour_outputs = self._nopre(pour_outputs, prefix)
    num_examples = pour_outputs['nums'].shape[0]
    example_dicts = []
    for row_num in xrange(num_examples):
      example_dict = {}

      nums = pour_outputs['nums'][row_num].flatten()
      example_dict['nums'] = feat._float_feat(nums)

      nats = pour_outputs['nats'][row_num].astype(int).flatten()
      example_dict['nats'] = feat._int_feat(nats)

      if pour_outputs['diff'].size:
        diff = pour_outputs['diff'][row_num].astype(np.int64)
      else:
        diff = np.zeros(nums.shape, dtype=np.int64)
      example_dict['diff'] = feat._int_feat(diff)

      example_dict = self._pre(example_dict, prefix)
      example_dicts.append(example_dict)

    return example_dicts

  def _parse_example_dicts(self, example_dicts, prefix=''):
    pour_outputs = {'nums': [], 'nats': [], 'diff': []}

    shape = self.input_shape[1:]
    for example_dict in example_dicts:
      nums = example_dict[self._pre('nums', prefix)].reshape(shape)
      pour_outputs['nums'].append(nums)

      nats = example_dict[self._pre('nats', prefix)].reshape(shape)
      pour_outputs['nats'].append(nats)

      diff = example_dict[self._pre('diff', prefix)].reshape(shape)
      pour_outputs['diff'].append(diff)

    pour_outputs = {
      'nums': np.stack(pour_outputs['nums']),
      'nats': np.stack(pour_outputs['nats']).astype(bool),
      'diff': np.stack(pour_outputs['diff']).astype(np.timedelta64),
    }

    pour_outputs = self._pre(pour_outputs, prefix)
    return pour_outputs

  def _feature_def(self, num_cols=1, prefix=''):
    feature_dict = {}
    size = np.prod(self.input_shape[1:])
    feature_dict['nums'] = tf.FixedLenFeature([size], tf.float32)
    feature_dict['nats'] = tf.FixedLenFeature([size], tf.int64)
    feature_dict['diff'] = tf.FixedLenFeature([size], tf.int64)

    feature_dict = self._pre(feature_dict, prefix)
    return feature_dict

  def _shape_def(self, prefix=''):
    shape_dict = {}
    shape_dict['nums'] = self.input_shape[1:]
    shape_dict['nats'] = self.input_shape[1:]
    shape_dict['diff'] = self.input_shape[1:]

    shape_dict = self._pre(shape_dict, prefix)
    return shape_dict

  def __len__(self):
    assert self.input_dtype is not None, ("Run calc_global_values before attempting to get the length.")
    return len(self.index_to_cat_val)
