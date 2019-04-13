import transform as n
import pandas as pd
import numpy as np


class SetTransform(n.Transform):
  """Class used to create mappings from raw set-like to vectorized many hot normalized data and vice versa.

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

  attribute_list = ['index_to_set_val', 'set_val_to_index', 'columns', 'key', 'other_set_vals', 'as_int']

  def _setattributes(self, df, columns, other_set_vals=None, key=None, as_int=False, **kwargs):
    """Set all the relevant attributes for this subclass. Called by the constructor for the Transform class.

    Parameters
    ----------
    df : pd.DataFrame
      The dataframe with all the data used to define the mappings.
    columns : list of strs
      The column names of all the relevant columns that make up the data to be taken from the dataframe

    """

    self.columns = columns
    self.other_set_vals = other_set_vals
    self.as_int = as_int
    if key is None:
      self.key = self.columns[0]

    uniques = []

    # Get all the unique category values
    def get_uniques(row):
      for column_name in columns:
        for set_val in row[column_name]:
          if type(set_val) is str:
            set_val = unicode(set_val)
          uniques.append(set_val)
    df.apply(get_uniques, axis=1)
    uniques = set(uniques)

    # Add in any additional category values
    if other_set_vals is not None:
      uniques = uniques | set(other_set_vals)

    # Create the mapping from category values to index in the vector and
    # vice versa
    self.index_to_set_val = sorted(uniques)
    self.set_val_to_index = {
      set_val: num for num, set_val in enumerate(self.index_to_set_val)
    }

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
    indices = []
    # value is not found in the mapping.
    for column_num, column_name in enumerate(self.columns):
      for set_val in row[column_name]:
        if set_val not in self.set_val_to_index:
          if verbose:
            print "WARNING:", self.key, "'s", set_val, "not in list of values.", sorted(self.set_val_to_index.keys())
          continue
        index = self.set_val_to_index[set_val]
        indices.append(index)

    # Convert the indices to a vector
    return self.indices_to_vector(indices)

  def indices_to_vector(self, indices):
    """Convert a the indices of categories (given by set_val_to_index) to a vector.

    Parameters
    ----------
    indices: list of ints
      The indices of each category value to be converted to a vector

    Returns
    -------
    np.array(
      shape=[len(self)],
      dtype=np.float64
    )
      The vectorized and normalized data.

    """
    # Create the many hot vectors by replacing the zeros with ones in the
    # index's location
    vector = np.zeros([len(self.set_val_to_index)], dtype=np.float64)
    for index in indices:
      if index == -1:
        continue
      vector[index] += 1.0
    if self.as_int:
      vector = vector.astype(np.int64)
    return vector

  def vector_to_indices(self, vector, verbose=True, cutoff=0.000001,):
    """Convert a vector into a list of indices corresponding to category values.

    Parameters
    ----------
    vector: np.array(
      shape=[len(self)],
      dtype=np.float64
    )
      The vectorized and normalized data.

    Returns
    -------
    list of ints
      The indices of each category value.

    """
    # Find all the locations where it's greater than zero.
    indices = np.where(vector > cutoff)[0]

    return indices.tolist()

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
    # Get the indices of the category values from the vector
    indices = self.vector_to_indices(vector, verbose)
    r_list = []

    # Fill a dictionary representing the original row. Convert the index to
    # a category value
    row = {c: None for c in self.columns}
    for index in indices:
      r_list.append(self.index_to_set_val[index])
    row[self.columns[0]] = sorted(r_list)

    # Conver the dict into a pandas series, representing the row.
    return pd.Series([row[c] for c in self.columns], index=self.columns)

  def input_summary(self, key, rows, verbose=True, tabs=0):
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
    # Get all the indices corresponding to the category values.
    indices = []
    for row in rows:
      for col in row:
        for set_val in col:
          indices.append(self.set_val_to_index[set_val])

    # Find the maximum index
    max_index = np.max(indices)

    # Create a histogram with each index getting it's own bin.
    hist_summary = so.summary_histogram(key, indices, bins=np.arange(max_index + 2), subtract_one=True)

    # Print out some summary statistics
    if verbose:
      print '\t'*tabs, '-'*50
      print '\t'*tabs, key, 'summary info:'
      print '\t'*tabs, '-'*50
      print '\t' * (tabs + 1), 'Index to category value:'
      for num, val in enumerate(self.index_to_set_val):
        print '\t' * (tabs + 1), num, val

      print
      print '\t' * (tabs + 1), 'Category value to index:'
      for key in sorted(self.set_val_to_index):
        print '\t' * (tabs + 1), key, self.set_val_to_index[key]

      counts = np.bincount(indices)
      sorted_indices = np.argsort(counts)[::-1]

      print
      print '\t' * (tabs + 1), 'Category values sorted by num occurences:'
      for index in sorted_indices:
        print '\t' * (tabs + 1), self.index_to_set_val[index], counts[index]
    return hist_summary
