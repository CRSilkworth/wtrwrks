"""NumTransform definition."""
import transform as n
import numpy as np
import warnings
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

  attribute_dict = {'norm_mode': None, 'norm_axis': None, 'fill_nan_func': None, 'name': '', 'mean': None, 'std': None, 'min': None, 'max': None, 'dtype': np.float64, 'input_dtype': None}

  def _extract_pour_outputs(self, tap_dict, prefix=''):
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
    return {self._pre(k, prefix): tap_dict[self._pre(k, prefix)] for k in ['nums', 'nans']}

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
    return funnel_dict[self._pre('IsNan_0/slots/a', prefix)]

  def _feature_def(self, num_cols=1, prefix=''):
    """Get the dictionary that contain the FixedLenFeature information for each key found in the example_dicts. Needed in order to build ML input pipelines that read tfrecords.

    Parameters
    ----------
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    dict
      The dictionary with keys equal to those that are found in the Transform's example dicts and values equal the FixedLenFeature defintion of the example key.

    """
    feature_dict = {}
    size = np.prod(self.input_shape[1:])
    feature_dict['nums'] = tf.FixedLenFeature([size], tf.float32)
    feature_dict['nans'] = tf.FixedLenFeature([size], tf.int64)

    feature_dict = self._pre(feature_dict, prefix)
    return feature_dict

  def _get_example_dicts(self, pour_outputs, prefix=''):
    """Create a list of dictionaries for each example from the outputs of the pour method.

    Parameters
    ----------
    pour_outputs : dict
      The outputs of the _extract_pour_outputs method.
    prefix : str
      Any additional prefix string/dictionary keys start with. Defaults to no additional prefix.

    Returns
    -------
    list of dicts of features
      The example dictionaries which contain tf.train.Features.

    """
    pour_outputs = self._nopre(pour_outputs, prefix)
    num_examples = pour_outputs['nums'].shape[0]
    example_dicts = []
    for row_num in xrange(num_examples):
      example_dict = {}

      # Flatten the arrays and convert them into features.
      nums = pour_outputs['nums'][row_num].flatten()
      example_dict['nums'] = feat._float_feat(nums)
      nans = pour_outputs['nans'][row_num].astype(int).flatten()
      example_dict['nans'] = feat._int_feat(nans)
      example_dict = self._pre(example_dict, prefix)
      example_dicts.append(example_dict)

    return example_dicts

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
    funnel_dict = {
      'Replace_0/slots/replace_with': self.fill_nan_func(array)
    }
    if array is not None:
      funnel_dict['IsNan_0/slots/a'] = array
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
    pour_outputs = self._nopre(pour_outputs, prefix)

    num_nans = len(np.where(pour_outputs['nans'])[0])
    tap_dict = {
      'nums': pour_outputs['nums'],
      'nans': pour_outputs['nans'],
      'replaced_vals': np.full([num_nans], np.nan, dtype=self.input_dtype),
      'Replace_0/tubes/replace_with_shape': (num_nans,),
    }
    # If there was a norm mode set then add in all the additional information.
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

  def _parse_example_dicts(self, example_dicts, prefix=''):
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
    pour_outputs = {'nums': [], 'nans': []}
    shape = self.input_shape[1:]

    # Go through each example dict, pull out the arrays associated with each
    # key of pour outputs, reshape them to their proper shape and add them
    # to individual lists so that they can be stacked together.
    for example_dict in example_dicts:
      nums = example_dict[self._pre('nums', prefix)].reshape(shape)
      pour_outputs['nums'].append(nums)

      nans = example_dict[self._pre('nans', prefix)].reshape(shape)
      pour_outputs['nans'].append(nans)

    # Stack the lists of arrays into arrays with a batch dimension
    pour_outputs = {
      'nums': np.stack(pour_outputs['nums']),
      'nans': np.stack(pour_outputs['nans']).astype(bool),
    }
    pour_outputs = self._pre(pour_outputs, prefix)
    return pour_outputs

  def _setattributes(self, **kwargs):
    """Set the actual attributes of the Transform and do some value checks to make sure they valid inputs.

    Parameters
    ----------
    **kwargs :
      The keyword arguments that set the values of the attributes defined in the attribute_dict.

    """
    super(NumTransform, self)._setattributes(**kwargs)

    if self.norm_mode not in (None, 'min_max', 'mean_std'):
      raise ValueError(self.norm_mode + " not a valid norm mode.")

    if self.fill_nan_func is None:
      self.fill_nan_func = lambda array: np.full(array[np.isnan(array)].shape, np.array(0))

  def _shape_def(self, prefix=''):
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
    shape_dict = {}
    shape_dict['nums'] = self.input_shape[1:]
    shape_dict['nans'] = self.input_shape[1:]

    shape_dict = self._pre(shape_dict, prefix)
    return shape_dict

  def calc_global_values(self, array, verbose=True):
    """Calculate all the values of the Transform that are dependent on all the examples of the dataset. (e.g. mean, standard deviation, unique category values, etc.) This method must be run before any actual transformation can be done.

    Parameters
    ----------
    array : np.ndarray
      The entire dataset.
    verbose : bool
      Whether or not to print out warnings.

    """
    # Set the input dtype
    self.input_dtype = array.dtype
    self.input_shape = array.shape
    array = array.astype(self.dtype)

    if self.norm_mode == 'mean_std':
      # Find the means and standard deviations of each column
      array[np.isnan(array)] = self.fill_nan_func(array)
      self.mean = np.mean(array, axis=self.norm_axis)
      self.std = np.std(array, axis=self.norm_axis)

      # If any of the standard deviations are 0, replace them with 1's and
      # print out a warning
      if (self.std == 0).any():
        if verbose:
          warnings.warn("NumTransform " + self.name + " has a zero-valued std, replacing with 1.")
        self.std[self.std == 0.] = 1.0

    elif self.norm_mode == 'min_max':
      # Find the means and standard deviations of each column
      array[np.isnan(array)] = self.fill_nan_func(array)
      self.min = np.min(array, axis=self.norm_axis)
      self.max = np.max(array, axis=self.norm_axis)

      # Test to make sure that min and max are not equal. If they are replace
      # with default values.
      if (self.min == self.max).any():
        if self.max.shape:
          self.max[self.max == self.min] = self.max[self.max == self.min] + 1
        else:
          self.max = self.max + 1

        if verbose:
          warnings.warn("NumTransform " + self.name + " the same values for min and max, replacing with " + str(self.min) + " " + str(self.max) + " respectively.")

  def define_waterwork(self, array=empty):
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
    nans, _ = td.isnan(array)
    nums, _ = td.replace(nans['a'], nans['target'])

    nums['replaced_vals'].set_name('replaced_vals')
    nums['mask'].set_name('nans')

    # Do any additional normalization
    if self.norm_mode == 'mean_std':
      nums, _ = nums['target'] - self.mean
      nums, _ = nums['target'] / self.std
    elif self.norm_mode == 'min_max':
      nums, _ = nums['target'] - self.min
      nums, _ = nums['target'] / (self.max - self.min)

    nums['target'].set_name('nums')
