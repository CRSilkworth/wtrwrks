import production.utils.dir_functions as d
import pandas as pd
import numpy as np
import tensorflow as tf
import production.transforms as n
import production.utils.timer as t
import copy_reg
import types
import multiprocessing as mp


def _pickle_method(method):
  func_name = method.im_func.__name__
  obj = method.im_self
  cls = method.im_class
  return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
  for cls in cls.mro():
    try:
      func = cls.__dict__[func_name]
    except KeyError:
      pass
    else:
      break
  return func.__get__(obj, cls)


copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class TransformSet(object):
  """Container class to hold all the transforms for a dataset to map from raw rows in a dataframe to vectorized, normalized data and vice versa.

  Parameters
  ----------
  from_file : str
    The path to the file to recreate the transform set from disk.

  Attributes
  ----------
  transforms : dict
    A dictionary whos values hold all the transforms for the dataset.
  indices : dict of tuples
    The beginning and end indices in the vector where each of the transforms writes to.
  classes : list of strs
    The class names of each transform
  keys : list of strs
    The key used to reference each transform
  vector_length : int
    The total length of the vector produced by the transform set.

  """

  def __init__(self, from_file=None):
    # Initalize the attributes
    self.transforms = {}
    self.indices = {}
    self.classes = {}
    self.keys = []
    self.vector_length = 0
    self.row_num_to_index = {}
    self.index_to_row_num = {}
    self.index_columns = {}

    # If a file path is given fill the attributes from the loaded dict.
    if from_file is not None:
      save_dict = d.read_from_file(from_file)
      self._from_save_dict(save_dict)

  def __setitem__(self, key, transform):
    """Add a transform to the transform set.

    Parameters
    ----------
    key : str
      The key to be used to look up the transform
    transform : Transform object
      The transform to be added.

    """
    # Add the key to the list of keys
    self.keys.append(key)

    # Save the class name
    self.classes[key] = transform.__class__.__name__

    # Save the transform
    self.transforms[key] = transform
    transform.key = key

    # define the first index to be the current vecto length. Have it go
    # until the length of the vector outputed by the transform.
    self.indices[key] = (self.vector_length, self.vector_length + len(transform))
    self.vector_length += len(transform)

  def add_row_num_mappings(self, key, df, index_column):
    """Add a mapping from an index in a dataframe to a unique row number.

    Parameters
    ----------
    key : str
      a key to lookup this particular mapping e.g. 'train' or 'validation'
    df : pd.DataFrame
      The dataframe to create the mappings from
    index_column : str
      The column to use as an index for the mapping to and from the dataframe. Must be unique for each row in the dataframe.

    """
    self.index_columns[key] = index_column
    self.row_num_to_index[key] = df[index_column].tolist()
    # Make sure that the values of index_column are unique
    assert len(set(self.row_num_to_index[key])) == len(self.row_num_to_index[key]), str(index_column) + " is not a unique identifier for the row"

    # Create the mappings to row number from index and vice versa
    self.index_to_row_num[key] = {index: row_num for row_num, index in enumerate(self.row_num_to_index[key])}

  def __getitem__(self, key):
    """Return the transform corresponding to key"""
    return self.transforms[key]

  def __iter__(self):
    """Iterator of the transform set is just the iterator of the transforms dictionary"""
    return iter(self.transforms)

  def row_index_to_row_num(self, key, row):
    row_index = row[self.index_columns[key]]
    row_num = self.index_to_row_num[key][row_index]

    return np.array(row_num, dtype=np.int64)

  def row_num_to_row_index(self, key, row_num):
    return self.row_num_to_index[key][int(row_num)]

  def row_to_vector(self, row, verbose=True):
    """Convert a row in a dataframe to a vector by calling all the various transforms' row_to_vector methods.

    Parameters
    ----------
    row : pd.Series
      A row in a dataframe where the index is the column name and the value is the column value.
    verbose : bool
      Whether or not to print out warnings.

    Returns
    -------
    np.array(
      shape=[self.vector_length],
      dtype=np.float64
    )
      The vectorized and normalized data.

    """
    # Go through each key, call the row_to_vector metho of each transform,
    # concatenate them into one vector.
    vectors = []
    for key in self.keys:
      transform = self.transforms[key]
      subvector = transform.row_to_vector(row)
      vectors.append(subvector)

    return np.concatenate(vectors, axis=0)

  def row_to_subvectors(self, row, verbose=True):
    """Convert a row in a dataframe to a dictionary of subvectors by calling all the various transforms' row_to_vector methods.

    Parameters
    ----------
    row : pd.Series
      A row in a dataframe where the index is the column name and the value is the column value.
    verbose : bool
      Whether or not to print out warnings.

    Returns
    -------
    dict
      a dictionary who's keys are the transform keys and the values are the subvectors created by each transform.

    """
    # Go through each key, call the row_to_vector metho of each transform,
    # add them to a dictionary.
    subvectors = {}
    for key in self.keys:
      transform = self.transforms[key]

      subvector = transform.row_to_vector(row, verbose)
      subvectors[key] = subvector

    return subvectors

  def subvectors_to_row(self, subvectors, verbose=True):
    """Convert a dictionary of subvectors into the corresponding row for the dataframe.

    Parameters
    ----------
    dict
      a dictionary who's keys are the transform keys and the values are the subvectors created by each transform.
    verbose : bool
      Whether or not to print out warnings.

    Returns
    -------
    row : pd.Series
      A row in a dataframe where the index is the column name and the value is the column value.

    """
    # Go through each of the transforms, and get a list of all the columns that need to be recreated.
    row = {}
    all_columns = []
    for key in self.keys:
      transform = self.transforms[key]

      for c in transform.columns:
        row[c] = None
        all_columns.append(c)

    # Recreate the columns for each of the transforms except for the time
    # series ones since they may need to information for their groupby
    # values.
    for key in self.keys:
      transform = self.transforms[key]
      cls = self.classes[key]

      if cls == 'TimeSeriesTransform':
        continue
      subvector = subvectors[key]

      sub_row = transform.vector_to_row(subvector, verbose)
      row.update(sub_row.to_dict())

    # Recreate the time series columns, using the groupby values where
    # applicable.
    for key in self.keys:
      transform = self.transforms[key]
      cls = self.classes[key]

      if cls != 'TimeSeriesTransform':
        continue

      subvector = subvectors[key]

      groupby_val = '__whole_dataset__'
      if transform.groupby is not None:
        groupby_val = row[transform.groupby]

      sub_row = transform.vector_to_row(subvector, verbose, groupby_val=groupby_val)

      row.update(sub_row.to_dict())

    # Create a pandas series from the dictionary, removing duplicate indices
    row = pd.Series([row[c] for c in all_columns], index=all_columns)
    row = row[~row.index.duplicated(keep='first')]
    return row

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
    subvectors = {}
    for key in self.keys:
      indices = self.indices[key]
      subvectors[key] = vector[indices[0]: indices[1]]

    return self.subvectors_to_row(subvectors, verbose)

  def batch_row_to_vector(self, rows, verbose=True):
    """Convert a dataframe to a batch of vectors.

    Parameters
    ----------
    row : pd.Series
      A row in a dataframe where the index is the column name and the value is the column value.
    verbose : bool
      Whether or not to print out warnings.

    Returns
    -------
    np.array(
      shape=[batch_size, len(self)],
      dtype=np.float64
    )
      The vectorized and normalized data.

    """
    vectors = []
    for row in rows:
      vectors.append(self.row_to_vector(row, verbose))

    return np.stack(vectors, axis=0)

  def batch_row_to_subvectors(self, rows, verbose=True, num_threads=1):
    """Convert a dataframe to a dictionary of subvectors by calling all the various transforms' row_to_vector methods.

    Parameters
    ----------
    rows : pd.DataFrame
      A dataframe where the index is the column name and the value is the column value.
    verbose : bool
      Whether or not to print out warnings.

    Returns
    -------
    list of dict
      a dictionary who's keys are the transform keys and the values are the subvectors created by each transform.

    """
    if num_threads != 1:
      pool = mp.Pool()
      list_of_subvectors = pool.map(self.row_to_subvectors, rows.iterrows())
      pool.close()
    else:
      list_of_subvectors = []

      def fill_list(row):
        list_of_subvectors.append(self.row_to_subvectors(row, verbose))
      rows.apply(fill_list, axis=1)

    batch_subvectors = {}
    for key in self.keys:
      batch_subvectors[key] = []
      for subvectors in list_of_subvectors:
        batch_subvectors[key].append(np.expand_dims(subvectors[key], axis=0))
      batch_subvectors[key] = np.concatenate(batch_subvectors[key], axis=0)
    return batch_subvectors

  def batch_subvectors_to_row(self, subvectors_dict, verbose=True, num_threads=1):
    """Convert a dictionary of batched subvectors into the corresponding dataframe.

    Parameters
    ----------
    dict
      a dictionary who's keys are the transform keys and the values are the subvectors created by each transform.
    verbose : bool
      Whether or not to print out warnings.

    Returns
    -------
    pd.DataFrame
      The raw dataframe corresponding to the batch of subvectors.

    """
    rows = []
    subvectors_list = [subvectors_dict[key] for key in self.keys]
    if num_threads != 1:
      pool = mp.Pool()

      list_of_subvectors = [
        {key: val for key, val in zip(self.keys, row_vals)} for row_vals in zip(*subvectors_list)
      ]
      rows = pool.map(self.subvectors_to_row, list_of_subvectors)
      pool.close()
    else:
      for row_vals in zip(*subvectors_list):
        subvectors = {key: val for key, val in zip(self.keys, row_vals)}
        row = self.subvectors_to_row(subvectors, verbose)
        rows.append(row)

    return pd.DataFrame(rows)

  def batch_vectors_to_row(self, vectors, verbose=True):
    """Convert the vectorized and normalized batched data back into it's raw dataframe (of the batch).

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
    pd.DataFrame
      The raw dataframe corresponding to the batch of vectors.

    """
    rows = []
    for vector in vectors:
      rows.append(self.vector_to_row(vector, verbose))

    return rows

  def save_to_file(self, path):
    """Save the transform set to disk.

    Parameters
    ----------
    path : str
      The path to save the file to.

    """
    save_dict = {}
    save_dict['keys'] = self.keys
    save_dict['vector_length'] = self.vector_length
    save_dict['transforms'] = {}
    save_dict['index_to_row_num'] = self.index_to_row_num
    save_dict['row_num_to_index'] = self.row_num_to_index
    for key in self.keys:
      save_dict['transforms'][key] = {
        'cls': self.classes[key],
        'norm_dict': self.transforms[key]._save_dict(),
        'indices': self.indices[key]
      }
    d.save_to_file(save_dict, path)

  def _from_save_dict(self, save_dict):
    """Short summary.

    Parameters
    ----------
    save_dict : dict
      The dictionary of all the attributes to recreate the transform set.

    """
    self.keys = save_dict['keys']
    self.vector_length = save_dict['vector_length']
    self.transforms = {}
    self.indices = {}
    self.classes = {}
    self.index_to_row_num = save_dict['index_to_row_num']
    self.row_num_to_index = save_dict['row_num_to_index']
    for key in self.keys:
      cls = save_dict['transforms'][key]['cls']
      indices = save_dict['transforms'][key]['indices']
      norm_dict = save_dict['transforms'][key]['norm_dict']

      self.transforms[key] = eval('n.' + cls)(save_dict=norm_dict)
      self.classes[key] = cls
      self.indices[key] = indices
