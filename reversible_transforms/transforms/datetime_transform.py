import transform as n
import pandas as pd
import numpy as np
import datetime
import warnings

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
  attribute_dict = {'col_index': None, 'norm_mode': None, 'fill_nan_func': None, 'name': '', 'mean': None, 'std': None, 'min': None, 'max': None, 'dtype': np.float64, 'input_dtype': None, 'zero_datetime': datetime.datetime(1970, 1, 1)}

  def _setattributes(self, **kwargs):
    super(DateTimeTransform, self)._setattributes(self.attribute_dict, **kwargs)

    if self.norm_mode not in (None, 'min_max', 'mean_std'):
      raise ValueError(self.norm_mode + " not a valid norm mode.")

    if type(self.zero_datetime) is datetime.datetime:
      self.zero_datetime = np.datetime64(self.zero_datetime)

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
    # Get the inputted dtype
    self.input_dtype = array.dtype

    # Pull out the relevant column
    col_array = array[:, self.col_index]

    if self.norm_mode == 'mean_std':
      # Find the means and standard deviations of each column
      temp_col_array = col_array[~np.isnat(col_array)]
      temp_col_array = (temp_col_array - self.zero_datetime)/np.timedelta64(1, 's')
      temp_col_array = temp_col_array.astype(self.dtype)

      if not len(temp_col_array):
        raise ValueError("Inputted col_array has no non nan values.")
      self.mean = np.mean(temp_col_array, axis=0)
      self.std = np.std(temp_col_array, axis=0)

      # If any of the standard deviations are 0, replace them with 1's and
      # print out a warning
      if self.std == 0:
        if verbose:
          warnings.warn("DatetimeTransform " + self.name + " has a zero-valued std, replacing with 1.")
        self.std += 1.0

    elif self.norm_mode == 'min_max':
      # Find the means and standard deviations of each column
      temp_col_array = col_array[~np.isnat(col_array)]
      temp_col_array = (temp_col_array - self.zero_datetime)/np.timedelta64(1, 's')
      temp_col_array = temp_col_array.astype(self.dtype)
      if not len(temp_col_array):
        raise ValueError("Inputted col_array has no non nan values.")

      self.min = np.min(temp_col_array, axis=0)
      self.max = np.max(temp_col_array, axis=0)

      # Test to make sure that min and max are not equal. If they are replace
      # with default values.
      if self.min == self.max:
        self.max = self.max + 1

        if verbose:
          warnings.warn("DatetimeTransform " + self.name + " the same values for min and max, replacing with " + str(self.min) + " " + str(self.max) + " respectively.")

  def forward_transform(self, array, verbose=True):
    """Convert a row in a dataframe to a vector.

    Parameters
    ----------
    row : pd.Series
      A row in a dataframe where the index is the column name and the value is the column value.
    verbose : bool
      Whether or not to print out warnings.

    Returns
    -------
    np.array(
      shape=[len(self)],
      dtype=np.float64
    )
      The vectorized and normalized data.

    """
    assert self.input_dtype is not None, ("Run calc_global_values before running the transform")

    col = array[:, self.col_index: self.col_index + 1]
    isnan = np.isnat(col)

    if self.fill_nan_func is not None:
      col = self.fill_nan_func(array, self.col_index)

    # Find the total seconds since the start time
    secs = (col - self.zero_datetime)/np.timedelta64(1, 's')
    secs = secs.astype(self.dtype)

    # Subtract out the mean and divide by the standard deviation to give a
    # mean of zero and standard deviation of one.
    if self.norm_mode == 'mean_std':
      secs = (secs - self.mean) / self.std
    elif self.norm_mode == 'min_max':
      secs = (secs - self.min) / (self.max - self.min)

    # Convert them to a vector
    return {'isnan': isnan, 'data': secs}

  def seconds_to_vector(self, seconds, verbose=True):
    """Convert the total seconds since start time to vectorized and normalized data.

    Parameters
    ----------
    seconds : list of numerical
      The seconds to be normalized and converted into the a vector.
    verbose : bool
      Whether or not to print out warnings.

    Returns
    -------
    np.array(
      shape=[len(self)],
      dtype=np.float64
    )
      The vectorized and normalized data.

    """
    # Create an array from the inputted seconds, subtract out the mean and
    # divide by the standard deviation giving a mean of zero and and
    # standard deviation of one.
    vector = np.array(seconds, dtype=np.float64)
    if self.mean_std:
      vector = (vector - self.means) / self.stds

    return vector

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
    assert self.input_dtype is not None, ("Run calc_global_values before running the transform")
    col = np.array(arrays_dict['data'], copy=True)
    col[arrays_dict['isnan']] = np.datetime64('NaT')

    # Undo the mean/std or min/max normalizations to give back the unscaled
    # values.
    if self.norm_mode == 'mean_std':
      col = col * self.std + self.mean
    elif self.norm_mode == 'min_max':
      col = col * (self.max - self.min) + self.min

    col = self.zero_datetime + col * np.timedelta64(1, 's')

    return col.astype(self.input_dtype)
