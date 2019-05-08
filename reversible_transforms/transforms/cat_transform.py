import transform as n
import reversible_transforms.waterworks.waterwork as wa
import reversible_transforms.waterworks.placeholder as pl
import reversible_transforms.tanks.tank_defs as td
import numpy as np
import warnings
import os


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

  attribute_dict = {'norm_mode': None, 'ignore_null': False, 'name': '', 'valid_cats': None, 'mean': None, 'std': None, 'dtype': np.float64, 'input_dtype': None, 'index_to_cat_val': None, 'cat_val_to_index': None}

  def _setattributes(self, **kwargs):
    super(CatTransform, self)._setattributes(self.attribute_dict, **kwargs)

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
      # Create one hot vectors for each row.
      col_array = array[np.isin(array, self.index_to_cat_val)]
      if not col_array.shape[0]:
        raise ValueError("Inputted col_array has no non null values.")

      one_hots = np.zeros([col_array.shape[0], len(uniques)], dtype=np.float64)
      row_nums = np.arange(col_array.shape[0], dtype=np.int64)

      indices = np.vectorize(self.cat_val_to_index.get)(col_array)
      one_hots[row_nums, indices] += 1

      # Find the means and standard deviation of the whole dataframe.
      self.mean = np.mean(one_hots, axis=0)
      self.std = np.std(one_hots, axis=0)

      # If there are any standard deviations of 0, replace them with 1's,
      # print out a warning.
      if len(self.std[self.std == 0]):
        zero_std_cat_vals = []
        for index in np.where(self.std == 0.0)[0]:
          zero_std_cat_vals.append(self.index_to_cat_val[index])

        if verbose:
          warnings.warn("WARNING: " + self.name + " has zero-valued stds at " + str(zero_std_cat_vals) + " replacing with 1's")

        self.std[self.std == 0] = 1.0

  def get_waterwork(self):
    assert self.input_dtype is not None, ("Run calc_global_values before running the transform")

    with wa.Waterwork(name=self.name) as ww:
      input = pl.Placeholder(np.ndarray, self.input_dtype)

      cti = td.cat_to_index(
        input,
        self.cat_val_to_index,
        type_dict={'cats': np.ndarray}
      )
      cti['missing_vals'].set_name('missing_vals')

      one_hots = td.one_hot(cti['target'], len(self.cat_val_to_index))

      if self.norm_mode == 'mean_std':
        one_hots = one_hots['target'] - self.mean
        one_hots = one_hots['target'] / self.std

      one_hots['target'].set_name('one_hots')

    return ww

  def pour(self, array):
    ww = self.get_waterwork()

    tank_name = os.path.join(self.name, 'CatToIndex_0')
    funnel_dict = {(tank_name, 'cats'): array[:, 0]}
    tap_dict = ww.pour(funnel_dict, key_type='str')

    return {k: tap_dict[k] for k in ['one_hots', 'missing_vals']}

  def pump(self, one_hots, missing_vals):
    ww = self.get_waterwork()

    mvs = -1.0 * np.ones([len(missing_vals)])

    if self.norm_mode == 'mean_std':
      tap_dict = {
        (self._name('OneHotTyped_0'), 'missing_vals'): mvs,
        (self._name('DivTyped_0'), 'target'): one_hots,
        (self._name('DivTyped_0'), 'smaller_size_array'): self.std,
        (self._name('DivTyped_0'), 'a_is_smaller'): False,
        (self._name('DivTyped_0'), 'missing_vals'): np.array([], dtype=float),
        (self._name('DivTyped_0'), 'remainder'): np.array([], dtype=one_hots.dtype),
        (self._name('SubTyped_0'), 'smaller_size_array'): self.mean,
        (self._name('SubTyped_0'), 'a_is_smaller'): False,
        (self._name('CatToIndex_0'), 'missing_vals'): missing_vals,
        (self._name('CatToIndex_0'), 'cat_to_index_map'): self.cat_val_to_index,
        (self._name('CatToIndex_0'), 'input_dtype'): self.input_dtype
      }
    else:
      tap_dict = {
        (self._name('OneHotTyped_0'), 'missing_vals'): mvs,
        (self._name('OneHotTyped_0'), 'target'): one_hots,
        (self._name('CatToIndex_0'), 'missing_vals'): missing_vals,
        (self._name('CatToIndex_0'), 'cat_to_index_map'): self.cat_val_to_index,
        (self._name('CatToIndex_0'), 'input_dtype'): self.input_dtype
      }
    funnel_dict = ww.pump(tap_dict)

    array_key = ww.get_slot(os.path.join(self.name, 'CatToIndex_0'), 'cats')
    return np.expand_dims(funnel_dict[array_key], axis=1)

  def __len__(self):
    assert self.input_dtype is not None, ("Run calc_global_values before attempting to get the length.")
    return len(self.index_to_cat_val)
