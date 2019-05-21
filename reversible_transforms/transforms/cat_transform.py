import transform as n
import reversible_transforms.waterworks.waterwork as wa
from reversible_transforms.waterworks.empty import empty
import reversible_transforms.tanks.tank_defs as td
import reversible_transforms.read_write.tf_features as feat
import numpy as np
import warnings
import os
import tensorflow as tf

class CatTransform(n.Transform):
  """Class used to create mappings from raw categorical to vectorized, normalized data and vice versa.

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

  attribute_dict = {'norm_mode': None, 'norm_axis': 0, 'ignore_null': False, 'name': '', 'valid_cats': None, 'mean': None, 'std': None, 'dtype': np.float64, 'input_dtype': None, 'index_to_cat_val': None, 'cat_val_to_index': None}

  def _setattributes(self, **kwargs):
    super(CatTransform, self)._setattributes(**kwargs)

    if self.norm_mode not in (None, 'mean_std'):
      raise ValueError(self.norm_mode + " not a valid norm mode.")

  def calc_global_values(self, array, verbose=True):
    """Set all the relevant attributes for this subclass. Called by the constructor for the Transform class.

    Parameters
    ----------
    df : pd.DataFrame
      The dataframe with all the data used to define the mappings.
    columns : list of strs
      The column names of all the relevant columns that make up the data to be taken from the dataframe

    """
    # Set the input dtype
    self.input_dtype = array.dtype

    # Pull out the relevant column

    # Get all the unique category values
    if self.valid_cats is not None:
      uniques = sorted(set(self.valid_cats))
    else:
      uniques = sorted(set(np.unique(array)))

    # If null are to be ignored then remove them.
    if self.ignore_null:
      if None in uniques:
        uniques.remove(None)
      if np.nan in uniques:
        uniques.remove(np.nan)

    # Create the mapping from category values to index in the vector and
    # vice versa
    self.index_to_cat_val = uniques
    self.cat_val_to_index = {}
    for unique_num, unique in enumerate(uniques):
      # if isinstance(unique, float) and np.isnan(unique):
      #   self.index_to_cat_val[unique_num] = None
      cat_val = self.index_to_cat_val[unique_num]
      self.cat_val_to_index[cat_val] = unique_num

    if self.norm_mode == 'mean_std':
      if not self.index_to_cat_val:
        raise ValueError("index_to_cat_val has no valid values.")
      # Create one hot vectors for each row.
      valid_cats = np.isin(array, self.index_to_cat_val)
      array = np.array(array, copy=True)

      default_val = self.index_to_cat_val[0]
      if isinstance(default_val, float) and np.isnan(default_val):
        default_val = self.index_to_cat_val[1]
      array[~valid_cats] = default_val

      one_hots = np.zeros(list(array.shape) + [len(uniques)], dtype=np.float64)

      indices = np.vectorize(self.cat_val_to_index.get)(array)
      one_hot_indices = np.unravel_index(np.arange(indices.size, dtype=np.int32), indices.shape)
      one_hot_indices = list(one_hot_indices) + [indices.flatten()]
      one_hot_indices = np.stack(one_hot_indices).astype(int)

      one_hots[one_hot_indices] = 1
      one_hots[~valid_cats] = 0

      self.mean = np.mean(one_hots, axis=self.norm_axis)
      self.std = np.std(one_hots, axis=self.norm_axis)

      # col_array = array[np.isin(array, self.index_to_cat_val)]
      # if not col_array.shape[0]:
      #   raise ValueError("Inputted col_array has no non null values.")
      #
      # one_hots = np.zeros(list(col_array.shape) + [len(uniques)], dtype=np.float64)
      # row_nums = np.arange(col_array.shape[0], dtype=np.int64)
      #
      # indices = np.vectorize(self.cat_val_to_index.get)(col_array)
      # one_hots[row_nums, indices] += 1
      # # Find the means and standard deviation of the whole dataframe.
      # self.mean = np.mean(one_hots, axis=0)
      # self.std = np.std(one_hots, axis=0)

      # If there are any standard deviations of 0, replace them with 1's,
      # print out a warning.
      if len(self.std[self.std == 0]):
        zero_std_cat_vals = []
        for index in np.where(self.std == 0.0)[0]:
          zero_std_cat_vals.append(self.index_to_cat_val[index])

        if verbose:
          warnings.warn("WARNING: " + self.name + " has zero-valued stds at " + str(zero_std_cat_vals) + " replacing with 1's")

        self.std[self.std == 0] = 1.0

  def define_waterwork(self, array=empty):
    cti, cti_slots = td.cat_to_index(
      array,
      self.cat_val_to_index,
    )
    cti['missing_vals'].set_name('missing_vals')

    cloned, _ = td.clone(cti['target'])
    cloned['a'].set_name('indices')

    one_hots, _ = td.one_hot(cloned['b'], len(self.cat_val_to_index))

    if self.norm_mode == 'mean_std':
      one_hots, _ = one_hots['target'] - self.mean
      one_hots, _ = one_hots['target'] / self.std

    one_hots['target'].set_name('one_hots')

  def _get_funnel_dict(self, array=None, prefix=''):
    funnel_name = self._pre('CatToIndex_0/slots/cats', prefix)
    funnel_dict = {funnel_name: array}

    return funnel_dict

  def _extract_pour_outputs(self, tap_dict, prefix=''):
    r_dict = {self._pre(k, prefix): tap_dict[self._pre(k, prefix)] for k in ['one_hots', 'missing_vals', 'indices']}
    return r_dict

  def _get_tap_dict(self, pour_outputs, prefix=''):
    pour_outputs = self._nopre(pour_outputs, prefix)
    mvs = -1.0 * np.ones([len(pour_outputs['missing_vals'])])
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
        'CatToIndex_0/tubes/cat_to_index_map': self.cat_val_to_index,
        'CatToIndex_0/tubes/input_dtype': self.input_dtype
      }
    else:
      tap_dict = {
        'OneHot_0/tubes/missing_vals': mvs,
        'one_hots': pour_outputs['one_hots'],
        'indices': pour_outputs['indices'],
        'missing_vals': pour_outputs['missing_vals'],
        'CatToIndex_0/tubes/cat_to_index_map': self.cat_val_to_index,
        'CatToIndex_0/tubes/input_dtype': self.input_dtype
      }
    return self._pre(tap_dict, prefix)

  def _extract_pump_outputs(self, funnel_dict, prefix=''):
    array_key = self._pre('CatToIndex_0/slots/cats', prefix)
    return funnel_dict[array_key]

  def _get_example_dicts(self, pour_outputs, prefix=''):
    pour_outputs = self._nopre(pour_outputs, prefix)
    mask = pour_outputs['indices'] == -1
    missing_vals = pour_outputs['missing_vals']
    full_missing_vals = self._full_missing_vals(mask, missing_vals)

    num_examples = pour_outputs['indices'].shape[0]
    example_dicts = []
    for row_num in xrange(num_examples):
      example_dict = {}

      one_hots = pour_outputs['one_hots'][row_num].flatten()
      example_dict['one_hots'] = feat._float_feat(one_hots)

      indices = pour_outputs['indices'][row_num].flatten()
      example_dict['indices'] = feat._int_feat(indices)

      missing_val = full_missing_vals[row_num]
      dtype = self.input_dtype
      if dtype.type in (np.string_, np.unicode_):
        example_dict['missing_vals'] = feat._bytes_feat(missing_val)
      elif dtype in (np.int32, np.int64):
        example_dict['missing_vals'] = feat._int_feat(missing_val)
      elif dtype in (np.float32, np.float64):
        example_dict['missing_vals'] = feat._float_feat(missing_val)
      else:
        raise TypeError("Only string and number types are supported. Got " + str(dtype))

      example_dict = self._pre(example_dict, prefix)
      example_dicts.append(example_dict)

    return example_dicts

  def _parse_example_dicts(self, example_dicts, prefix=''):
    pour_outputs = {'one_hots': [], 'indices': []}

    missing_vals = []
    for example_dict in example_dicts:
      pour_outputs['one_hots'].append(example_dict[self._pre('one_hots', prefix)])
      pour_outputs['indices'].append(example_dict[self._pre('indices', prefix)])
      missing_vals.append(example_dict[self._pre('missing_vals', prefix)])

    pour_outputs = {
      'one_hots': np.stack(pour_outputs['one_hots']),
      'indices': np.stack(pour_outputs['indices']),
    }

    missing_vals = np.stack(missing_vals)
    missing_vals = missing_vals[pour_outputs['indices'] == -1]
    pour_outputs['missing_vals'] = missing_vals.tolist()

    pour_outputs = self._pre(pour_outputs, prefix)
    return pour_outputs

  def _feature_def(self, num_cols=1, prefix=''):
    dtype = self.input_dtype
    if dtype.type in (np.string_, np.unicode_):
      tf_dtype = tf.string
    elif dtype in (np.int32, np.int64):
      tf_dtype = tf.int64
    elif dtype in (np.float32, np.float64):
      tf_dtype = tf.float32
    else:
      raise TypeError("Only string and number types are supported. Got " + str(dtype))

    shape = len(self)

    feature_dict = {}
    feature_dict['missing_vals'] = tf.FixedLenFeature([1], tf_dtype)
    feature_dict['one_hots'] = tf.FixedLenFeature(shape, tf.float32)
    feature_dict['indices'] = tf.FixedLenFeature([1], tf.int64)

    feature_dict = self._pre(feature_dict, prefix)
    return feature_dict

  def __len__(self):
    assert self.input_dtype is not None, ("Run calc_global_values before attempting to get the length.")
    return len(self.index_to_cat_val)
