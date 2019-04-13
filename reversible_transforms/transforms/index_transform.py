import transform as n
import reversible_transforms.utils.dir_functions as d
import pandas as pd
import numpy as np
import pprint


class IndexTransform(n.Transform):
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

  attribute_list = ['index_to_cat_val', 'cat_val_to_index', 'columns', 'key', 'other_cat_vals']

  def _setattributes(self, df, columns, other_cat_vals=None, key=None, **kwargs):
    """Set all the relevant attributes for this subclass. Called by the constructor for the Transform class.

    Parameters
    ----------
    df : pd.DataFrame
      The dataframe with all the data used to define the mappings.
    columns : list of strs
      The column names of all the relevant columns that make up the data to be taken from the dataframe

    """
    assert len(columns) == 1, ("columns and only be a list of size one for IndexTransform.")

    self.columns = columns
    self.other_cat_vals = other_cat_vals
    self.key = key
    self.means = None
    self.stds = None

    if key is None:
      self.key = self.columns[0]

    # Get all the unique category values
    uniques = set()
    uniques = uniques | set(df[columns[0]].unique())

    # Add in any additional category values
    if other_cat_vals is not None:
      uniques = uniques | set(other_cat_vals)

    # Create the mapping from category values to index in the vector and
    # vice versa
    self.index_to_cat_val = []
    for unique in sorted(uniques):
      if isinstance(unique, float) and np.isnan(unique):
        self.index_to_cat_val.append(None)
      else:
        self.index_to_cat_val.append(unique)

    self.cat_val_to_index = {
      cat_val: num for num, cat_val in enumerate(self.index_to_cat_val)
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
    # Pull out each of the category values.
    cat_val = row[self.columns[0]]

    # Find the indices for each category, filling with -1 if the category
    # value is not found in the mapping.
    index = self.nan_safe_cat_val_to_index(cat_val, verbose=verbose)

    # Convert the indices to a vector
    return np.array([index]).astype(np.int64)

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
    index = vector[0]

    # Fill a dictionary representing the original row. Convert the index to
    # a category value
    row = {self.columns[0]: self.index_to_cat_val[index]}

    # Conver the dict into a pandas series, representing the row.
    return pd.Series([row[c] for c in self.columns], index=self.columns)

  def nan_safe_cat_val_to_index(self, cat_val, verbose=True):
    """Convert a category value to it's corresponding index while mapping nans to None.

    Parameters
    ----------
    cat_val : hashable
      The category value to map to index
    verbose : bool
      Whether or not to print out booleans

    Returns
    -------
    int
      The corresponding index of the category value

    """
    # If the category value is in the dictionary, use it to map to index
    if cat_val in self.cat_val_to_index:
      index = self.cat_val_to_index[cat_val]
    # If the category value is s a nan, use the None mapping.
    elif isinstance(cat_val, float) and np.isnan(cat_val) and None in self.cat_val_to_index:
      index = self.cat_val_to_index[None]
    # Otherwise return -1
    else:
      if verbose:
        print "WARNING:", self.key, "'s", cat_val, "not in list of values.", sorted(self.cat_val_to_index.keys())
      index = -1

    return index

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
      for cat_val in row:
        index = self.nan_safe_cat_val_to_index(cat_val, verbose=verbose)
        indices.append(index)

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
      for num, val in enumerate(self.index_to_cat_val):
        print '\t' * (tabs + 1), num, val

      print
      print '\t' * (tabs + 1), 'Category value to index:'
      for key in sorted(self.cat_val_to_index):
        print '\t' * (tabs + 1), key, self.cat_val_to_index[key]

      counts = np.bincount(indices)
      sorted_indices = np.argsort(counts)[::-1]

      print
      print '\t' * (tabs + 1), 'Category values sorted by num occurences:'
      for index in sorted_indices:
        print '\t' * (tabs + 1), self.index_to_cat_val[index], counts[index]
    return hist_summary
  def __len__(self):
    return 1
