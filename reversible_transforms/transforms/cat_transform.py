import transform as n
import reversible_transforms.utils.dir_functions as d
import reversible_transforms.waterworks.waterwork as wa
import reversible_transforms.tanks.cat_to_index as ci
import reversible_transforms.tanks.one_hot as oh
import reversible_transforms.tanks.sub as su
import reversible_transforms.tanks.div as dv
import pandas as pd
import numpy as np
import pprint
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
      cti = ci.cat_to_index(
        None,
        self.cat_val_to_index,
        type_dict={'cats': np.ndarray}
      )

      one_hots = oh.one_hot(cti['target'], len(self.cat_val_to_index))

      if self.norm_mode == 'mean_std':
        sub = su.sub(one_hots['target'], self.mean)
        one_hots = dv.div(
          sub['target'], self.std
        )
    return ww

  def pour(self, array):
    ww = self.get_waterwork()

    tank_name = os.path.join(self.name, 'CatToIndexNP_0')
    funnel_dict = {(tank_name, 'cats'): array[:, 0]}
    tap_dict = ww.pour(funnel_dict, tuple_keys=True)

    print "HERE"
    if self.norm_mode == 'mean_std':
      output_dict = {
        'one_hots': tap_dict[('cat/DivBasicTyped_0', 'target')],
        'missing_vals': tap_dict[('cat/CatToIndexNP_0', 'missing_vals')]
      }
    else:
      output_dict = {
        'one_hots': tap_dict[('cat/OneHotNP_0', 'target')],
        'missing_vals': tap_dict[('cat/CatToIndexNP_0', 'missing_vals')]
      }

    return output_dict

  def pump(self, one_hots, missing_vals):
    ww = self.get_waterwork()

    mvs = -1.0 * np.ones([len(missing_vals)])

    if self.norm_mode == 'mean_std':
      tap_dict = {
        ('cat/OneHotNP_0', 'missing_vals'): mvs,
        ('cat/DivBasicTyped_0', 'target'): one_hots,
        ('cat/DivBasicTyped_0', 'b'): self.std,
        ('cat/DivBasicTyped_0', 'b'): self.std,
        ('cat/SubNP_0', 'smaller_size_array'): self.mean,
        ('cat/SubNP_0', 'a_is_smaller'): False,
        ('cat/CatToIndexNP_0', 'missing_vals'): missing_vals,
        ('cat/CatToIndexNP_0', 'cat_to_index_map'): self.cat_val_to_index,
        ('cat/CatToIndexNP_0', 'input_dtype'): self.input_dtype
      }
    else:
      tap_dict = {
        ('cat/OneHotNP_0', 'missing_vals'): mvs,
        ('cat/OneHotNP_0', 'target'): one_hots,
        ('cat/CatToIndexNP_0', 'missing_vals'): missing_vals,
        ('cat/CatToIndexNP_0', 'cat_to_index_map'): self.cat_val_to_index,
        ('cat/CatToIndexNP_0', 'input_dtype'): self.input_dtype
      }

    funnel_dict = ww.pump(tap_dict)

    array_key = ww.get_slot('cat/CatToIndexNP_0', 'cats')
    return np.expand_dims(funnel_dict[array_key], axis=1)

  # def pour(self, array, verbose=True):
  #   """Convert a row in a dataframe to a vector.
  #
  #   Parameters
  #   ----------
  #   row : pd.Series
  #     A row in a dataframe where the index is the column name and the value is the column value.
  #   verbose : bool
  #     Whether or not to print out warnings.
  #
  #   Returns
  #   -------
  #   np.array(
  #     shape=[len(self)],
  #     dtype=np.float64
  #   )
  #     The vectorized and normalized data.
  #
  #   """
  #   assert self.input_dtype is not None, ("Run calc_global_values before running the transform")
  #   self.temp_verbose = verbose
  #   # Find the indices for each category, filling with -1 if the category
  #   # value is not found in the mapping.
  #   map_to_indices = np.vectorize(self.nan_safe_cat_val_to_index)
  #   indices = map_to_indices(array[:, self.col_range[0]])
  #
  #   row_num = np.arange(array.shape[0])
  #   two_indices = np.stack([np.arange(array.shape[0]), indices], axis=1)
  #   two_indices = two_indices[two_indices[:, 1] != -1]
  #
  #   data = np.zeros(
  #     shape=[array.shape[0], len(self.cat_val_to_index)],
  #     dtype=self.dtype
  #   )
  #   data[row_num, indices] = 1
  #   data[indices == -1] = 0
  #
  #   if self.norm_mode == 'mean_std':
  #     data = (data - self.mean)/self.std
  #
  #   return {'data': data, 'cat_val': array[:, self.col_range[0]: self.col_range[1]], 'index': np.expand_dims(indices, axis=1)}
  #
  # def pump(self, arrays_dict, verbose=True):
  #   """Convert the vectorized and normalized data back into it's raw dataframe row.
  #
  #   Parameters
  #   ----------
  #   vector : np.array(
  #     shape=[len(self)],
  #     dtype=np.float64
  #   )
  #     The vectorized and normalized data.
  #   verbose : bool
  #     Whether or not to print out warnings.
  #
  #   Returns
  #   -------
  #   row : pd.Series
  #     A row in a dataframe where the index is the column name and the value is the column value.
  #
  #   """
  #   assert self.input_dtype is not None, ("Run calc_global_values before running the transform")
  #
  #   one_hot = arrays_dict['data']
  #
  #   # Get the indices of the category values from the vector
  #   if self.norm_mode == 'mean_std':
  #     one_hot = one_hot * self.std + self.mean
  #   # elif self.norm_mode == 'min_max':
  #   #   one_hot = one_hot * (self.max - self.min) + self.min
  #
  #   # Find all the locations where it's greater than zero.
  #   indices = np.argmax(one_hot, axis=1)
  #   get_cat_val = np.vectorize(lambda i: self.index_to_cat_val[i])
  #
  #   array = get_cat_val(indices)
  #   array = np.expand_dims(array, axis=1)
  #   not_found_indices = arrays_dict['index'] == -1
  #   array[not_found_indices] = arrays_dict['cat_val'][not_found_indices]
  #
  #   # Convert the dict into numpy array
  #   return array.astype(self.input_dtype)
  #
  # def nan_safe_cat_val_to_index(self, cat_val, verbose=True):
  #   """Convert a category value to it's corresponding index while mapping nans to None.
  #
  #   Parameters
  #   ----------
  #   cat_val : hashable
  #     The category value to map to index
  #   verbose : bool
  #     Whether or not to print out booleans
  #
  #   Returns
  #   -------
  #   int
  #     The corresponding index of the category value
  #
  #   """
  #   # If the category value is in the dictionary, use it to map to index
  #   if cat_val in self.cat_val_to_index:
  #     index = self.cat_val_to_index[cat_val]
  #
  #   # If the category value is s a nan, use the None mapping.
  #   elif isinstance(cat_val, float) and np.isnan(cat_val) and None in self.cat_val_to_index:
  #     index = self.cat_val_to_index[None]
  #
  #   # Otherwise return -1
  #   else:
  #     if hasattr(self, 'temp_verbose') and self.temp_verbose:
  #       warnings.warn("CatTransform " + self.name + "'s " + str(cat_val) + " not in list of values." + str(sorted(self.cat_val_to_index.keys())))
  #     index = -1
  #   return index
  #
  # def __len__(self):
  #   assert self.input_dtype is not None, ("Run calc_global_values before attempting to get the length.")
  #   return len(self.index_to_cat_val)
