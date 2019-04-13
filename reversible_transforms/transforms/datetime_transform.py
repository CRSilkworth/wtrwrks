import transform as n
import pandas as pd
import numpy as np
import datetime


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

  attribute_list = ['index_to_column_name', 'column_name_to_index', 'means', 'stds', 'start_datetimes', 'columns', 'key', 'mean_std']

  def _setattributes(self, df, columns, start_datetimes, key=None, mean_std=True, **kwargs):
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
    self.mean_std = mean_std
    self.means = None
    self.stds = None
    if key is None:
      self.key = columns[0]

    # Default verbose to true and look for it in kwargs to set.
    verbose = True
    if 'verbose' in kwargs:
      verbose = kwargs['verbose']

    # Save the inputted attributes
    self.columns = columns
    self.start_datetimes = start_datetimes

    # Create the mappings from column name to index in the vector and vice
    # versa
    self.index_to_column_name = columns
    self.column_name_to_index = {key: val for val, key in enumerate(columns)}

    if self.mean_std:
      # Calculate the means and standard deviations of the total seconds
      # from the start times.
      self.means = np.zeros(shape=[len(columns)], dtype=np.float64)
      self.stds = np.zeros(shape=[len(columns)], dtype=np.float64)
      for column_num, column_name in enumerate(columns):
        ts = df[column_name]
        ts = (ts - start_datetimes[column_num]).dt.total_seconds()

        self.means[column_num] = np.mean(ts)
        self.stds[column_num] = np.std(ts)

      # If any of the standard deviations are 0 then replace them with 1's
      # and print out a warning.
      if len(self.stds[self.stds == 0]):
        zero_std_column_names = [columns[i] for i in np.where(self.stds == 0)]
        if verbose:
          print "WARNING: zero-valued stds, replacing with 1's:", zero_std_column_names
        self.stds[self.stds == 0] = 1.0

  def row_to_vector(self, row, verbose=True):
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
    # Get all the raw values
    vals = [row[c] for c in self.columns]

    # Find the total seconds since the start time
    seconds = []
    for column_num, val in enumerate(vals):
      s = (val - self.start_datetimes[column_num]).total_seconds()
      seconds.append(s)

    # Convert them to a vector
    return self.seconds_to_vector(seconds, verbose)

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

  def vector_to_seconds(self, vector, verbose=True):
    """Convert the vectorized and normalized data back into it's raw seconds.

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
    list of numerical
      The seconds since the start times.

    """
    # Undo the standard deviation and mean transformations to give back the original means and standard deviations.
    if self.mean_std:
      seconds = vector * self.stds + self.means
    return seconds.tolist()

  def vector_to_row(self, vector, verbose=True):
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
    # Conver the vector into a list of seconds since the start time.
    seconds = self.vector_to_seconds(vector, verbose)

    # Recreate the row from the seconds by findind the proper datetime and
    # creating a pandas series
    row = {}
    for s_num, (s, column) in enumerate(zip(seconds, self.columns)):
      row[column] = self.start_datetimes[s_num] + pd.to_timedelta(str(s) + 's')

    return pd.Series([row[c] for c in self.columns], index=self.columns)

  def input_summary(self, key, rows, verbose=True, tabs=0, bins=100):
    """Create a summary and print out some summary statistics of the the whole dataframe given by rows.

    Parameters
    ----------
    key : str
      A key used to identify the data in tensorboard.
    rows : pd.DataFrame
      The dataframe with all the data to be summarized.
    verbose : bool
      Whether or not to print out the summary statistics.
    tabs : int (default 0)
      Number of tabs to indent the summary statistics.

    Returns
    -------
    tf.Summary
      The summary object used to create the tensorboard histogram.

    """
    # Create a histogram of all the seconds' values
    datetime_ints = (rows - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    hist_summary = so.summary_histogram(key, datetime_ints, bins=bins)

    # Print out some summary statistics
    if verbose:
      print '\t'*tabs, '-'*50
      print '\t' * tabs, key, 'summary info:'
      print '\t'*tabs, '-'*50
      print '\t' * (tabs + 1), 'min:', np.min(rows)
      print '\t' * (tabs + 1), 'max:', np.max(rows)
      median = datetime.datetime.utcfromtimestamp(np.median(datetime_ints))
      print '\t' * (tabs + 1), 'median:', median
      mean = datetime.datetime.utcfromtimestamp(np.mean(datetime_ints))
      print '\t' * (tabs + 1), 'mean:', mean

    return hist_summary
