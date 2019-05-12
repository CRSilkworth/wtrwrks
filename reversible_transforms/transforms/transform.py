import pandas as pd
import reversible_transforms.utils.dir_functions as d
import reversible_transforms.waterworks.waterwork as wa
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
  attribute_dict = {}
  
  def __init__(self, from_file=None, save_dict=None, **kwargs):

    if from_file is not None:
      save_dict = d.read_from_file(from_file)
      self._from_save_dict(save_dict)
    elif save_dict is not None:
      self._from_save_dict(save_dict)
    else:
      self._setattributes(**kwargs)

  def _setattributes(self, **kwargs):
    attribute_set = set(self.attribute_dict)
    invalid_keys = sorted(set(kwargs.keys()) - attribute_set)

    if invalid_keys:
      raise ValueError("Keyword arguments: " + str(invalid_keys) + " are invalid.")

    for key in self.attribute_dict:
      if key in kwargs:
        setattr(self, key, kwargs[key])
      else:
        setattr(self, key, self.attribute_dict[key])

  def define_waterwork(self):
    raise NotImplementedError()

  def get_waterwork(self):
    assert self.input_dtype is not None, ("Run calc_global_values before running the transform")

    with wa.Waterwork(name=self.name) as ww:
      self.define_waterwork()

    return ww

  def _add_name(self, string, prefix=''):
    return os.path.join(prefix, self.name, string)

  def _add_name_to_dict(self, d, prefix=''):
    r_d = {}
    for key in d:
      if type(key) is tuple and type(key[0]) in (str, unicode):
        r_d[(os.path.join(prefix, self.name, key[0]), key[1])] = d[key]
      elif type(key) in (str, unicode):
        r_d[os.path.join(prefix, self.name, key)] = d[key]
      else:
        r_d[key] = d[key]
    return r_d

  def pour(self, array, **kwargs):
    ww = self.get_waterwork()
    funnel_dict = self._get_funnel_dict(array, **kwargs)
    tap_dict = ww.pour(funnel_dict, key_type='str')
    return self._extract_pour_outputs(tap_dict)

  def pump(self, **kwargs):
    ww = self.get_waterwork()
    tap_dict = self._get_tap_dict(**kwargs)
    funnel_dict = ww.pump(tap_dict, key_type='str')
    return self._extract_pump_outputs(funnel_dict)

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
