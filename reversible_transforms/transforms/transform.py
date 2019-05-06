import pandas as pd
import reversible_transforms.utils.dir_functions as d
import os

class Transform(object):
  """Abstract class used to create mappings from raw to vectorized, normalized data and vice versa.

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
    The list of attributes that need to be saved in order to fully reconstruct the transform object. Must start with the attribute which has the same length as the outputed vector from row_to_vector.

  """

  def __init__(self, from_file=None, save_dict=None, **kwargs):

    if from_file is not None:
      save_dict = d.read_from_file(from_file)
      self._from_save_dict(save_dict)
    elif save_dict is not None:
      self._from_save_dict(save_dict)
    else:
      self._setattributes(**kwargs)

  def _setattributes(self, attribute_dict, **kwargs):
    attribute_set = set(attribute_dict)
    invalid_keys = sorted(set(kwargs.keys()) - attribute_set)

    if invalid_keys:
      raise ValueError("Keyword arguments: " + str(invalid_keys) + " are invalid.")

    for key in attribute_dict:
      if key in kwargs:
        setattr(self, key, kwargs[key])
      else:
        setattr(self, key, attribute_dict[key])

  def row_to_vector(self, row):
    """Abstract method to be defined by the subclasses."""
    pass

  def vector_to_row(self, vector):
    """Abstract method to be defined by the subclasses."""
    pass

  def batch_vector_to_row(self, vectors):
    """Recreate a partial raw dataframe from batch of vectors.

    Parameters
    ----------
    vectors : np.array(
      shape=[batch_size, len(self)],
      dtype=np.float64
    )
      The vectors to be mapped back into a dataframe.

    Returns
    -------
    pd.DataFrame
      The 'raw' data corresponding to the batch of vectors. Where 'raw' means it was first mapped to vectors and then mapped back.

    """
    rows = []
    for vector in vectors:
      row = self.vector_to_row(vector)
      rows.append(row)
    return pd.DataFrame(rows)

  def _name(self, tank_name):
    return os.path.join(self.name, tank_name)

  def _save_dict(self):
    """Create the dictionary of values needed in order to reconstruct the transform."""
    save_dict = {}
    for key in self.attribute_dict:
      save_dict[key] = getattr(self, key)
    return save_dict

  def _from_save_dict(self, save_dict):
    """Reconstruct the transform object from the dictionary of attributes."""
    for key in self.attribute_dict:
      setattr(self, key, save_dict[key])

  def save_to_file(self, path):
    """Save the transform object to disk."""
    save_dict = self._save_dict()
    d.save_to_file(save_dict, path)

  def __len__(self):
    """Get the length of the vector outputted by the row_to_vector method."""
    return len(getattr(self, self.attribute_list[0]))

  def __str__(self):
    """Return the stringified values for each of the attributes in attribute list."""
    return str({a: str(getattr(self, a)) for a in self.attribute_dict})
