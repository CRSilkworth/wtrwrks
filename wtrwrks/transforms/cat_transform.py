"""Definition of the CatTransform."""
import transform as n
from wtrwrks.waterworks.empty import empty
import wtrwrks.tanks.tank_defs as td
import wtrwrks.read_write.tf_features as feat
import numpy as np
import logging
import tensorflow as tf


class CatTransform(n.Transform):
  """Class used to create mappings from raw categorical to vectorized, normalized data and vice versa.

  Parameters
  ----------
  name : str
    The name of the transform.
  from_file : str
    The path to the saved file to recreate the transform object that was saved to disk.
  save_dict : dict
    The dictionary to recreate the transform object
  index_to_cat_val : list
    The mapping from index number to category value.
  cat_val_to_index : dict
    The mapping from category value to index number.


  Attributes
  ----------
  input_dtype: numpy dtype
    The datatype of the original inputted array.
  input_shape: list of ints
    The shape of the original inputted array.

  """

  attribute_dict = {'norm_mode': None, 'norm_axis': 0, 'name': '', 'mean': None, 'std': None, 'dtype': np.float64, 'input_dtype': None, 'input_shape': None, 'index_to_cat_val': None, 'cat_val_to_index': None}
  attribute_dict.update({k: v for k, v in n.Transform.attribute_dict.iteritem() if k not in attribute_dict})

  required_params = set(['index_to_cat_val'])
  required_params.update(n.Transform.required_params)
  def __init__(self, from_file=None, save_dict=None, **kwargs):
    super(CatTransform, self).__init__(from_file, save_dict, **kwargs)

    if len(self.index_to_cat_val) != len(set(self.index_to_cat_val)):
      raise ValueError("All elements of index_to_cat_val must be unique")

  def __len__(self):
    """Get the length of the vector outputted by the row_to_vector method."""
    assert self.input_dtype is not None, ("Run calc_global_values before attempting to get the length.")
    return len(self.index_to_cat_val)

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
    r_dict = {self._pre(k, prefix): tap_dict[self._pre(k, prefix)] for k in ['one_hots', 'missing_vals', 'indices']}
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
    array_key = self._pre('CatToIndex_0/slots/cats', prefix)
    return funnel_dict[array_key]

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
    funnel_name = self._pre('CatToIndex_0/slots/cats', prefix)
    funnel_dict = {funnel_name: array}

    return funnel_dict

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
    pour_outputs = super(CatTransform, self)._get_tap_dict(pour_outputs, prefix)
    pour_outputs = self._nopre(pour_outputs, prefix)
    mvs = -1 * np.ones(pour_outputs['indices'].shape)
    dtype = pour_outputs['one_hots'].dtype
    if self.norm_mode == 'mean_std':
      tap_dict = {
        'OneHot_0/tubes/missing_vals': mvs,
        'one_hots': pour_outputs['one_hots'],
        'indices': pour_outputs['indices'],
        'Div_0/tubes/smaller_size_array': self.std,
        'Div_0/tubes/a_is_smaller': False,
        'Div_0/tubes/missing_vals': np.array([], dtype=float),
        'Div_0/tubes/remainder': np.array([], dtype=dtype),
        'Sub_0/tubes/smaller_size_array': self.mean,
        'Sub_0/tubes/a_is_smaller': False,
        'missing_vals': pour_outputs['missing_vals'],
        'CatToIndex_0/tubes/input_dtype': self.input_dtype
      }
    else:
      tap_dict = {
        'OneHot_0/tubes/missing_vals': mvs,
        'one_hots': pour_outputs['one_hots'],
        'indices': pour_outputs['indices'],
        'missing_vals': pour_outputs['missing_vals'],
        'CatToIndex_0/tubes/input_dtype': self.input_dtype
      }
    return self._pre(tap_dict, prefix)

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
    att_dict['missing_vals'] = {
      'shape': list(self.input_shape[1:]),
      'tf_type': feat.select_tf_dtype(self.input_dtype),
      'size': feat.size_from_shape(self.input_shape[1:]),
      'feature_func': feat.select_feature_func(self.input_dtype),
      'np_type': self.input_dtype
    }
    one_hots_shape = list(self.input_shape[1:]) + [len(self.index_to_cat_val)]
    att_dict['one_hots'] = {
      'shape': one_hots_shape,
      'tf_type': feat.select_tf_dtype(self.dtype),
      'size': feat.size_from_shape(one_hots_shape),
      'feature_func': feat.select_feature_func(self.dtype),
      'np_type': self.dtype
    }
    att_dict['indices'] = {
      'shape': list(self.input_shape[1:]),
      'tf_type': tf.int64,
      'size': feat.size_from_shape(self.input_shape[1:]),
      'feature_func': feat._int_feat,
      'np_type': np.int64
    }

    att_dict = self._pre(att_dict, prefix)
    return att_dict

  def _start_calc(self):
    # Create the mapping from category values to index in the vector and
    # vice versa
    self.cat_val_to_index = {}
    for unique_num, unique in enumerate(self.index_to_cat_val):
      cat_val = self.index_to_cat_val[unique_num]
      self.cat_val_to_index[cat_val] = unique_num

    self.num_examples = 0.

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

  def _calc_global_values(self, array):
    """Calculate all the values of the Transform that are dependent on all the examples of the dataset. (e.g. mean, standard deviation, unique category values, etc.) This method must be run before any actual transformation can be done.

    Parameters
    ----------
    array : np.ndarray
      The entire dataset.
    """

    batch_size = float(array.shape[0])
    total_examples = self.num_examples + batch_size

    if self.norm_mode == 'mean_std':
      if not self.index_to_cat_val:
        raise ValueError("index_to_cat_val has no valid values.")

      # Find all the category values which are not known
      valid_cats = np.isin(array, self.index_to_cat_val)
      array = np.array(array, copy=True)

      # Replace the unknown category values with some default value.
      default_val = self.index_to_cat_val[0]
      if isinstance(default_val, float) and np.isnan(default_val):
        default_val = self.index_to_cat_val[1]
      array[~valid_cats] = default_val

      one_hots = np.zeros(list(array.shape) + [len(self.index_to_cat_val)], dtype=np.float64)

      # Convert the category values to indices, using the cat_vat_to_index.
      indices = np.vectorize(self.cat_val_to_index.get)(array)

      # Generate all the indices of the 'indices' array. i.e. one_hot_indices is
      # equal to something like (0, 0...0), (0, 0 ... 1)... (n1, n2... nk).
      # Where n1..nk are the size of each dimension of 'indices'. Then append
      # the corresponding value from 'indices' to get the full location of where
      # a 1 should be in the one_hots array.
      one_hot_indices = np.unravel_index(np.arange(indices.size, dtype=np.int32), indices.shape)
      one_hot_indices = list(one_hot_indices) + [indices.flatten()]

      # Set all the proper locations to 1. And then undo the setting of the
      # not valid categories.
      one_hots[one_hot_indices] = 1
      one_hots[~valid_cats] = 0

      # Get the mean and standard deviation of the one_hots.
      if self.mean is None:
        self.mean = np.mean(one_hots, axis=self.norm_axis)
        self.var = np.var(one_hots, axis=self.norm_axis)
      else:
        self.mean = (self.num_examples / total_examples) * self.mean + (batch_size / total_examples) * np.mean(one_hots, axis=self.norm_axis)
        self.var = np.var(one_hots, axis=self.norm_axis)

    self.num_examples += batch_size


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
    # Convert the category values to indices.
    cti, cti_slots = td.cat_to_index(
      array,
      self.cat_val_to_index,
    )
    cti['missing_vals'].set_name('missing_vals')

    # Clone the indices so that a copy of 'indices' can be outputted as a tap.
    cloned, _ = td.clone(cti['target'])
    cloned['a'].set_name('indices')

    # Convert the indices into one-hot vectors.
    one_hots, _ = td.one_hot(cloned['b'], len(self.cat_val_to_index))

    # Normalize the one_hots if the norm_mode is set.
    if self.norm_mode == 'mean_std':
      one_hots, _ = one_hots['target'] - self.mean
      one_hots, _ = one_hots['target'] / self.std

    one_hots['target'].set_name('one_hots')

    if return_tubes is not None:
      ww = one_hots['target'].waterwork
      r_tubes = []
      for r_tube_key in return_tubes:
        r_tubes.append(ww.maybe_get_tube(r_tube_key))
      return r_tubes
