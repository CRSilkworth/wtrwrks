import transform as n
import pandas as pd
import numpy as np
import warnings

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

  attribute_dict = {'col_index': None, 'norm_mode': None, 'fill_null_func': None, 'name': '', 'mean': None, 'std': None, 'min': None, 'max': None, 'dtype': np.float64}

  def _setattributes(self, **kwargs):
    super(NumTransform, self)._setattributes(self.attribute_dict, **kwargs)

    if self.norm_mode not in (None, 'min_max', 'mean_std'):
      raise ValueError(self.norm_mode + " not a valid norm mode.")
    if self.col_index is None:
      raise ValueError("Must specify a column index.")

  def calc_global_values(self, array, verbose=True):
    """Set all the relevant attributes for this subclass that are needed in order to do the final transformation on the individual values of the col_array. e.g. find the mean/std for the mean/std normalization. Null values will be ignored during this step.

    Parameters
    ----------
    col_array : np.array(
      shape=[num_examples],
      dtype=self.dtype
    )
      The numpy with all the data used to define the mappings.
    verbose : bool
      Whether or not to print out warnings, supplementary info, etc.

    """
    # Pull out the relevant column
    col_array = array[:, self.col_index]

    if self.norm_mode == 'mean_std':
      # Find the means and standard deviations of each column
      temp_col_array = col_array[~np.isnan(col_array)]
      if not len(temp_col_array):
        raise ValueError("Inputted col_array has no non null values.")
      self.mean = np.mean(temp_col_array, axis=0).astype(self.dtype)
      self.std = np.std(temp_col_array, axis=0).astype(self.dtype)

      # If any of the standard deviations are 0, replace them with 1's and
      # print out a warning
      if self.std == 0:
        if verbose:
          warnings.warn("NumTransform " + self.name + " has a zero-valued std, replacing with 1.")
        self.std += 1.0

    elif self.norm_mode == 'min_max':
      # Find the means and standard deviations of each column
      temp_col_array = col_array[~np.isnan(col_array)]
      if not len(temp_col_array):
        raise ValueError("Inputted col_array has no non null values.")
      self.min = np.min(temp_col_array, axis=0).astype(self.dtype)
      self.max = np.max(temp_col_array, axis=0).astype(self.dtype)

      # Test to make sure that min and max are not equal. If they are replace
      # with default values.
      if self.min == self.max:
        if self.min > 0:
          self.min = self.min - self.min
        elif self.min == 0:
          self.min = self.min - self.min
          self.max = self.max - self.max + 1
        else:
          self.max = self.max - self.max

        if verbose:
          warnings.warn("NumTransform " + self.name + " the same values for min and max, replacing with " + self.min + " " + self.max + " respectively.")

  def forward_transform(self, row_array, row_index=None, verbose=True):
    """Convert a row in a dataframe to a vector.

    Parameters
    ----------
    row_array : pd.Series
      A row in a dataframe where the index is the column name and the value is the column value.
    verbose : bool
      Whether or not to print out warnings.

    Returns
    -------
    np.array(
      shape=[len(self)],
      dtype=np.float64
    )
      Description of returned object.

    """
    # Pull out each column value, put them in an array.
    val = row_array[self.col_index: self.col_index + 1].astype(self.dtype)
    is_null = np.zeros((1,), dtype=np.bool)
    if np.isnan(val).any():
      is_null = np.ones((1,), dtype=np.bool)
      if self.fill_null_func is not None:
        val = self.fill_null_func(row_array, self.col_index, row_index)

    # Subtract out the mean and divide by the standard deviation to give a
    # mean of zero and standard deviation of one.
    if self.norm_mode == 'mean_std':
      val = (val - self.mean) / self.std
    elif self.norm_mode == 'min_max':
      val = (val - self.min) / (self.max - self.min)

    return {'is_null': is_null, 'val': val}

  def backward_transform(self, arrays_dict, verbose=True):
    """Convert the vectorized and normalized data back into it's raw dataframe row.

    Parameters
    ----------
    vector : np.array(
      shape=[len(self)],
      dtype=np.float64
    )
      The vectorized and normalized data.
    verbose : bool
      Whether or not to print out warnings.

    Returns
    -------
    row : pd.Series
      A row in a dataframe where the index is the column name and the value is the column value.

    """
    if (arrays_dict['is_null'] == True).any():
      nan = np.ones((1,), dtype=self.dtype)
      nan[:] = np.nan
      return nan

    val = arrays_dict['val']

    # Undo the mean/std or min/max normalizations to give back the unscaled
    # values.
    if self.norm_mode == 'mean_std':
      val = val * self.std + self.mean
    elif self.norm_mode == 'min_max':
      val = val * (self.max - self.min) + self.min

    return val
