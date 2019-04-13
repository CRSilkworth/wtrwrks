import transform as n
import reversible_transforms.utils.dir_functions as d
import pandas as pd
import numpy as np
import tensorflow as tf


class TimeSeriesTransform(n.Transform):
  """Class used to create mappings from raw numerical time series data to vectorized, normalized data and vice versa.

  Parameters
  ----------
  df : pd.DataFrame
    The dataframe with all the data used to define the mappings.
  columns : list of strs
    The column names of all the relevant columns that make up the data to be taken from the dataframe
  groupby: str
    The column name to group the dataframe by before finding the mean and standard deviation in order to normalize the data. Defaults to grouping everything as one group.
  from_file : str
    The path to the saved file to recreate the transform object that was saved to disk.
  save_dict : dict
    The dictionary to rereate the transform object

  Attributes
  ----------
  attribute_list : list of strs
    The list of attributes that need to be saved in order to fully reconstruct the transform object.

  """

  attribute_list = ['index_to_column_name', 'column_name_to_index', 'means', 'stds', 'groupby', 'columns', 'key', 'mean_std']

  def _setattributes(self, df, columns, groupby=None, key=None, mean_std=True, **kwargs):
    """Set all the relevant attributes for this subclass. Called by the constructor for the Transform class.

    Parameters
    ----------
    df : pd.DataFrame
      The dataframe with all the data used to define the mappings.
    columns : list of strs
      The column names of all the relevant columns that make up the data to be taken from the dataframeself.
    groupby: str
      The column name to group the dataframe by before finding the mean and standard deviation in order to normalize the data. Defaults to grouping everything as one group.

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
    self.groupby = groupby

    # Create the mappings from column name to index in the vector and vice
    # versa
    self.index_to_column_name = columns
    self.column_name_to_index = {key: val for val, key in enumerate(columns)}

    if self.mean_std:
      # Divide up the dataframe by the column given by groupby and find
      # the mean/standard deviation for each group.
      means = {}
      stds = {}
      means['__whole_dataset__'] = np.mean(df[columns].values)
      stds['__whole_dataset__'] = np.std(df[columns].values)
      if groupby is not None:
        groups = df.groupby(groupby)
        for group_id, group in groups:
          means[group_id] = np.mean(group[columns].values)
          stds[group_id] = np.std(group[columns].values)

      # Replace any zero standard deviations with one and print out a
      # warning.
      for key in stds:
        if stds[key] == 0:
          if verbose:
            print "WARNING: zero-valued stds, replacing with 1's:", key
          stds[key] = 1.0

      self.means = means
      self.stds = stds

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
      Description of returned object.

    """
    # Get the value of the groupby column (if applicable)
    if self.groupby is not None:
      groupby_val = row[self.groupby]
    else:
      groupby_val = '__whole_dataset__'
    # Pull out the raw values of the rows
    vals = [row[c] for c in self.columns]
    vector = np.array(vals, dtype=np.float64)

    if self.mean_std:
      # If the groupby value isn't found print out a warning, and do
      # nothing to the raw values
      if groupby_val not in self.means:
        if verbose:
          print "WARNING:", self.key, "'s", groupby_val, "not in list of known values."
        groupby_val = '__whole_dataset__'

      # Convert the values to an array, subtract off the group mean and
      # divide by the group standard deviation to give a group of mean
      # zero and standard deviation of one.

      vector = (vector - self.means[groupby_val]) / self.stds[groupby_val]

    return vector

  def vector_to_row(self, vector, verbose=True, groupby_val='__whole_dataset__'):
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
    groupby_val: type(row[self.groupby])
      The value of the category used to group the data by before finding the mean/std.

    Returns
    -------
    row : pd.Series
      A row in a dataframe where the index is the column name and the value is the column value.

    """
    if self.mean_std:
      # If the groupby value isn't found, use the whole dataset's means
      if groupby_val not in self.means:
        if verbose:
          print "WARNING:", self.key, "'s", groupby_val, "not in list of known values."
        groupby_val = '__whole_dataset__'

      # Undo the standard deviation and mean transformations to give back
      # the original means and standard deviations.
      vector = vector * self.stds[groupby_val] + self.means[groupby_val]

    # Recreate the row by creating a pandas series with all the relevant columns filled in with their original raw values.
    row = {}
    for column_num, column in enumerate(self.columns):
      row[column] = vector[column_num]

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
    # Create a histogram fo all the values.
    hist_summary = so.summary_histogram(key, rows, bins=bins)

    # Print out some summary statistics.
    if verbose:

      print '\t'*tabs, '-'*50
      print '\t' * tabs, key, 'summary info:'
      print '\t'*tabs, '-'*50
      print '\t' * (tabs + 1), 'min, max:', np.min(rows), np.max(rows)
      print '\t' * (tabs + 1), 'median:', np.median(rows)
      print '\t' * (tabs + 1), 'mean:', np.mean(rows)
      print '\t' * (tabs + 1), 'std:', np.std(rows)

      print '\t' * (tabs + 1), 'mins by time range:'
      print '\t' * (tabs + 1), np.min(rows, axis=0)
      print '\t' * (tabs + 1), 'maxes by time range:'
      print '\t' * (tabs + 1), np.max(rows, axis=0)
      print '\t' * (tabs + 1), 'medians by time :'
      print '\t' * (tabs + 1), np.median(rows, axis=0)

    return hist_summary
