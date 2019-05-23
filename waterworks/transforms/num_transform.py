import transform as n
import numpy as np
import warnings
import waterworks.tanks.tank_defs as td
import waterworks.read_write.tf_features as feat
import numpy as np
import tensorflow as tf
from waterworks.waterworks.empty import empty
import os

class NumTransform(n.Transform):
  """Class used to create mappings from raw numerical data to vectorized, normalized data and vice versa.

  Parameters
  ----------
  df : pd.DataFrame
    The dataframe with all the data used to define the mappings.
  columns : list of strs
    The column names of all the relevant columns that make up the data to be taken from the dataframe
  from_file : str
    The path to the saved file to recreate the transform object that was saved to disk.
  save_dict : dict
    The dictionary to rereate the transform object

  Attributes
  ----------
  attribute_list : list of strs
    The list of attributes that need to be saved in order to fully reconstruct the transform object.

  """

  attribute_dict = {'norm_mode': None, 'norm_axis': None, 'fill_nan_func': None, 'name': '', 'mean': None, 'std': None, 'min': None, 'max': None, 'dtype': np.float64, 'input_dtype': None}

  def _setattributes(self, **kwargs):
    super(NumTransform, self)._setattributes(**kwargs)

    if self.norm_mode not in (None, 'min_max', 'mean_std'):
      raise ValueError(self.norm_mode + " not a valid norm mode.")

    if self.fill_nan_func is None:
      self.fill_nan_func = lambda array: np.full(array[np.isnan(array)].shape, np.array(0))

  def calc_global_values(self, array, verbose=True):
    """Set all the attributes which use global information for this subclass that are needed in order to do the final transformation on the individual values of the col_array. e.g. find the mean/std for the mean/std normalization. Null values will be ignored during this step.

    Parameters
    ----------
    array : np.array(
      shape=[num_examples, total_input_dim],
      dtype=self.dtype
    )
      The numpy with all the data needed to define the mappings.
    verbose : bool
      Whether or not to print out warnings, supplementary info, etc.

    """
    # Set the input dtype
    self.input_dtype = array.dtype
    self.input_shape = array.shape
    array = array.astype(self.dtype)

    if self.norm_mode == 'mean_std':
      # Find the means and standard deviations of each column
      self.mean = np.nanmean(array, axis=self.norm_axis)
      self.std = np.nanstd(array, axis=self.norm_axis)

      # If any of the standard deviations are 0, replace them with 1's and
      # print out a warning
      if (self.std == 0).any():
        if verbose:
          warnings.warn("NumTransform " + self.name + " has a zero-valued std, replacing with 1.")
        self.std[self.std == 0.] = 1.0

    elif self.norm_mode == 'min_max':
      # Find the means and standard deviations of each column
      self.min = np.nanmin(array, axis=self.norm_axis)
      self.max = np.nanmax(array, axis=self.norm_axis)

      # Test to make sure that min and max are not equal. If they are replace
      # with default values.
      if (self.min == self.max).any():
        self.max[self.max == self.min] = self.max[self.max == self.min] + 1

        if verbose:
          warnings.warn("NumTransform " + self.name + " the same values for min and max, replacing with " + str(self.min) + " " + str(self.max) + " respectively.")

  def define_waterwork(self, array=empty):
    # Replace all the NaT's with the inputted replace_with.
    nans, _ = td.isnan(array)
    nums, _ = td.replace(nans['a'], nans['target'])

    nums['replaced_vals'].set_name('replaced_vals')
    nums['mask'].set_name('nans')

    if self.norm_mode == 'mean_std':
      nums, _ = nums['target'] - self.mean
      nums, _ = nums['target'] / self.std
    elif self.norm_mode == 'min_max':
      nums, _ = nums['target'] - self.min
      nums, _ = nums['target'] / (self.max - self.min)

    nums['target'].set_name('nums')

  def _get_funnel_dict(self, array=None, prefix=''):
    funnel_dict = {
      'Replace_0/slots/replace_with': self.fill_nan_func(array)
    }
    if array is not None:
      funnel_dict['IsNan_0/slots/a'] = array
    return self._pre(funnel_dict, prefix)

  def _extract_pour_outputs(self, tap_dict, prefix=''):
    return {self._pre(k, prefix): tap_dict[self._pre(k, prefix)] for k in ['nums', 'nans']}

  def _get_tap_dict(self, pour_outputs, prefix=''):
    pour_outputs = self._nopre(pour_outputs, prefix)

    num_nans = len(np.where(pour_outputs['nans'])[0])
    tap_dict = {
      'nums': pour_outputs['nums'],
      'nans': pour_outputs['nans'],
      'replaced_vals': np.full([num_nans], np.nan, dtype=self.input_dtype),
      'Replace_0/tubes/replace_with_shape': (num_nans,),
    }
    if self.norm_mode == 'mean_std' or self.norm_mode == 'min_max':
      if self.norm_mode == 'mean_std':
        sub_val = self.mean
        div_val = self.std
      else:
        sub_val = self.min
        div_val = self.max - self.min
      norm_mode_dict = {
        ('Sub_0/tubes/smaller_size_array'): sub_val,
        ('Sub_0/tubes/a_is_smaller'): False,
        ('Div_0/tubes/smaller_size_array'): div_val,
        ('Div_0/tubes/a_is_smaller'): False,
        ('Div_0/tubes/remainder'): np.array([], dtype=self.input_dtype),
        ('Div_0/tubes/missing_vals'): np.array([], dtype=float)
      }
      tap_dict.update(norm_mode_dict)
    return self._pre(tap_dict, prefix)

  def _extract_pump_outputs(self, funnel_dict, prefix=''):
    return funnel_dict[self._pre('IsNan_0/slots/a', prefix)]

  def _get_example_dicts(self, pour_outputs, prefix=''):
    pour_outputs = self._nopre(pour_outputs, prefix)
    num_examples = pour_outputs['nums'].shape[0]
    example_dicts = []
    for row_num in xrange(num_examples):
      example_dict = {}

      nums = pour_outputs['nums'][row_num].flatten()
      example_dict['nums'] = feat._float_feat(nums)

      nans = pour_outputs['nans'][row_num].astype(int).flatten()
      example_dict['nans'] = feat._int_feat(nans)

      example_dict = self._pre(example_dict, prefix)
      example_dicts.append(example_dict)

    return example_dicts

  def _parse_example_dicts(self, example_dicts, prefix=''):
    pour_outputs = {'nums': [], 'nans': []}
    shape = self.input_shape[1:]
    for example_dict in example_dicts:
      nums = example_dict[self._pre('nums', prefix)].reshape(shape)
      pour_outputs['nums'].append(nums)

      nans = example_dict[self._pre('nans', prefix)].reshape(shape)
      pour_outputs['nans'].append(nans)

    pour_outputs = {
      'nums': np.stack(pour_outputs['nums']),
      'nans': np.stack(pour_outputs['nans']).astype(bool),
    }
    pour_outputs = self._pre(pour_outputs, prefix)
    return pour_outputs

  def _feature_def(self, num_cols=1, prefix=''):

    feature_dict = {}
    size = np.prod(self.input_shape[1:])
    feature_dict['nums'] = tf.FixedLenFeature([size], tf.float32)
    feature_dict['nans'] = tf.FixedLenFeature([size], tf.int64)

    feature_dict = self._pre(feature_dict, prefix)
    return feature_dict

  def _shape_def(self, prefix=''):
    shape_dict = {}
    shape_dict['nums'] = self.input_shape[1:]
    shape_dict['nans'] = self.input_shape[1:]

    shape_dict = self._pre(shape_dict, prefix)
    return shape_dict
