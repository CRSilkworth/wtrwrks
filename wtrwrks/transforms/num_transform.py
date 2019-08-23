"""NumTransform definition."""
import transform as n
import numpy as np
import logging
import wtrwrks.tanks.tank_defs as td
import wtrwrks.read_write.tf_features as feat
import tensorflow as tf
from wtrwrks.waterworks.empty import empty


class NumTransform(n.Transform):
  """Class used to create mappings from raw numerical data to vectorized, normalized data and vice versa.

  Parameters
  ----------
  name : str
    The name of the transform.
  from_file : str
    The path to the saved file to recreate the transform object that was saved to disk.
  save_dict : dict
    The dictionary to recreate the transform object
  norm_mode : str, either 'mean_std' or 'max_min'
    The method used to normalize the data. Can either use 'mean_std' which subtracts out the mean and divides by the standard deviation or 'min_max' which subtracts out the min and divides by the (max - min).
  norm_axis : int, tuple or None
    The axis along which to calculate the mean, std, min, or max.
  fill_nan_func : func
    A function that takes in an array and replaces any nan values with some number. Defaults to a function that just replaces them with zero.

  Attributes
  ----------
  input_dtype : numpy dtype
    The datatype of the original inputted array.
  input_shape : list of ints
    The shape of the original inputted array.
  mean : np.ndarray
    The array of means used in the normalization.
  std : np.ndarray
    The array of standard deviations used in the normalization.
  min : np.ndarray
    The array of mins used in the normalization.
  max : np.ndarray
    The array of maxes used in the normalization.

  """

  attribute_dict = {'norm_mode': None, 'norm_axis': None, 'fill_nan_func': lambda array: np.array(0), 'name': '', 'mean': None, 'std': None, 'min': None, 'max': None}

  for k, v in n.Transform.attribute_dict.iteritems():
    if k in attribute_dict:
      continue
    attribute_dict[k] = v

  required_params = set([])
  required_params.update(n.Transform.required_params)

  def __init__(self, from_file=None, save_dict=None, **kwargs):
    super(NumTransform, self).__init__(from_file, save_dict, **kwargs)

    valid_norm_modes = ('mean_std', 'min_max', None)
    if self.norm_mode not in valid_norm_modes:
      raise ValueError("{} is an invalid norm_mode. Accepted norm mods are ".format(self.norm_mode, valid_norm_modes))

    valid_norm_axis = (0, 1, (0, 1), None)
    if self.norm_axis not in valid_norm_axis:
      raise ValueError("{} is an invalid norm_axis. Accepted norm axes are ".format(self.norm_axis, valid_norm_axis))

  def __len(self):
    # assert self.is_calc_run, ("Must run calc_global_values before taking the len.")
    return len(self.cols)

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
    att_dict['nums'] = {
      'shape': list([len(self.cols)]),
      'tf_type': feat.select_tf_dtype(self.dtype),
      'size': feat.size_from_shape([len(self.cols)]),
      'feature_func': feat.select_feature_func(self.dtype),
      'np_type': self.dtype
    }
    att_dict['nans'] = {
      'shape': [len(self.cols)],
      'tf_type': tf.int64,
      'size': feat.size_from_shape([len(self.cols)]),
      'feature_func': feat._int_feat,
      'np_type': np.bool
    }

    att_dict = self._pre(att_dict, prefix)
    return att_dict

  def _finish_calc(self):
    if self.norm_mode == 'mean_std':
      self.std = np.sqrt(self.var)
      # If there are any standard deviations of 0, replace them with 1's,
      # print out a warning.
      if len(self.std[self.std == 0]):
        zero_std_cat_vals = []
        for index in np.where(self.std == 0.0)[0]:
          zero_std_cat_vals.append(self.index_to_cat_val[index])

        logging.warn(self.name + " has zero-valued stds at " + str(zero_std_cat_vals) + " replacing with 1's")

        self.std[self.std == 0] = 1.0
    elif self.norm_mode == 'min_max':
      # Test to make sure that min and max are not equal. If they are replace
      # with default values.
      if (self.min == self.max).any():
        if self.max.shape:
          self.max[self.max == self.min] = self.max[self.max == self.min] + 1
        else:
          self.max = self.max + 1

        logging.warn("NumTransform " + self.name + " the same values for min and max, replacing with " + str(self.min) + " " + str(self.max) + " respectively.")

  def _calc_global_values(self, array, verbose=True):
    """Calculate all the values of the Transform that are dependent on all the examples of the dataset. (e.g. mean, standard deviation, unique category values, etc.) This method must be run before any actual transformation can be done.

    Parameters
    ----------
    array : np.ndarray
      The entire dataset.
    verbose : bool
      Whether or not to print out warnings.

    """
    if self.input_dtype is None:
      self.input_dtype = array.input_dtype
    else:
      array = np.array(array, dtype=self.input_dtype)

    if self.dtype is None:
      self.dtype = self.input_dtype

    batch_size = float(array.shape[0])
    total_examples = self.num_examples + batch_size

    array = array.astype(self.dtype)

    if self.norm_mode == 'mean_std':
      # Find the means and standard deviations of each column
      array[np.isnan(array)] = self.fill_nan_func(array)
      self.mean = np.mean(array, axis=self.norm_axis)
      self.std = np.std(array, axis=self.norm_axis)

      if self.mean is None:
        self.mean = np.mean(array, axis=self.norm_axis)
        self.var = np.var(array, axis=self.norm_axis)
      else:
        self.mean = (self.num_examples / total_examples) * self.mean + (batch_size / total_examples) * np.mean(array, axis=self.norm_axis)
        self.var = np.var(array, axis=self.norm_axis)

    elif self.norm_mode == 'min_max':
      # Find the means and standard deviations of each column
      array[np.isnan(array)] = self.fill_nan_func(array)

      if self.min is None:
        self.min = np.min(array, axis=self.norm_axis)
        self.max = np.max(array, axis=self.norm_axis)
      else:
        self.min = np.minimum(self.min, np.min(array, axis=self.norm_axis))
        self.max = np.maximum(self.max, np.max(array, axis=self.norm_axis))

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
    # Replace all the NaN's with the inputted replace_with function.
    nans, nans_slots = td.isnan(array)
    nans_slots['a'].set_name('array')

    nums, _ = td.replace(
      nans['a'], nans['target'],
      slot_plugs={
        'replace_with': lambda z: self.fill_nan_func(z[self._pre('array', prefix)])
      },
      tube_plugs={
        'replace_with': np.array([]),
        'replaced_vals': np.array(np.nan)
      }
    )

    nums['replaced_vals'].set_name('replaced_vals')
    nums['mask'].set_name('nans')

    # Do any additional normalization
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
