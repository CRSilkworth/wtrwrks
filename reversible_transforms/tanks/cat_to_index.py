import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import numpy as np


def cat_to_index(cats, cat_to_index_map, type_dict=None, waterwork=None, name=None):
  """Convert categorical values to index values according to cat_to_index_map. Handles the case where the categorical value is not in cat_to_index by mapping to -1.

  Parameters
  ----------
  cats : int, str, unicode, flot, numpy array or None
      The categorical values to be mapped to indices
  cat_to_index_map : dictionary
      A mapping from categorical values to indices
  type_dict : dict({
    keys - ['cats', 'cat_to_index_map']
    values - type of argument 'cats' type of argument 'cat_to_index_map'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).
  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  type_dict = ut.infer_types(type_dict, cats=cats, cat_to_index_map=cat_to_index_map)

  if type_dict['cat_to_index_map'] is not dict:
    raise TypeError("cat_to_index_map must be of type dict.")

  if type_dict['cats'] in (int, str, unicode, float):
    return CatToIndexBasic(cats=cats, cat_to_index_map=cat_to_index_map, waterwork=waterwork, name=name)
  elif type_dict['cats'] is np.ndarray:
    return CatToIndexNP(cats=cats, cat_to_index_map=cat_to_index_map, waterwork=waterwork, name=name)
  else:
    raise TypeError(type_dict['cats'] + "is not supported.")


class CatToIndex(ta.Tank):
  """The base CatToIndex class. All subclasses must have the same outputs where they output 'target' and 'missing_vals' and the cat_to_index_map. The cat_to_index_map must be a dict.

  Attributes
  ----------
  slot_keys : list of str
    The tank's (operation's) argument keys. They define the names of the inputs to the tank.
  tube_dict : dict(
    keys - strs. The tank's (operation's) output keys. THey define the names of the outputs of the tank
    values - types. The types of the arguments outputs.
  )
    The tank's (operation's) output keys and their corresponding types.

  """

  slot_keys = ['cats', 'cat_to_index_map']
  tube_dict = {
    'target': None,
    'missing_vals': list,
    'cat_to_index_map': dict
  }


class CatToIndexBasic(CatToIndex):
  """The CatToIndex class where the cats input is an int, str, unicode or float.

  Attributes
  ----------
  slot_keys : list of str
    The tank's (operation's) argument keys. They define the names of the inputs to the tank.
  tube_dict : dict(
    keys - strs. The tank's (operation's) output keys. THey define the names of the outputs of the tank
    values - types. The types of the arguments outputs.
  )
    The tank's (operation's) output keys and their corresponding types.


  """

  tube_dict = {
    'target': int,
    'missing_vals': list,
    'cat_to_index_map': dict
  }

  def _pour(self, cats, cat_to_index_map):
    """Execute the mapping in the pour (forward) direction .

    Parameters
    ----------
    cats : int, str, unicode, float
      The categorical values to be mapped to indices
    cat_to_index_map : dict
      The map from categorical values to indices.

    Returns
    -------
    dict(
      'target': int, float, other non array type
        The result of the sum of 'a' and 'b'.
      'missing_vals': list
        The list of all the cats that were not found in cat_to_index_map.
      'cat_to_index_map' : dict
        The map from categorical values to indices.
    )

    """
    # If the index is outside the allowed cats, save it in the
    # missing_vals. Otherwise convert the cat into an index. And keep
    # missing_vals empty.
    if cats in cat_to_index_map:
      target = cat_to_index_map[cats]
      missing_vals = []
    else:
      target = -1
      missing_vals = [cats]

    return {'target': target, 'missing_vals': missing_vals, 'cat_to_index_map': cat_to_index_map}

  def _pump(self, target, missing_vals, cat_to_index_map):
    """Execute the mapping in the pump (backward) direction .

    Parameters
    ----------
    target: int, float, other non array type
      The result of the sum of 'a' and 'b'.
    missing_vals: list
      The list of all the cats that were not found in cat_to_index_map.
    cat_to_index_map : dict
      The map from categorical values to indices.


    Returns
    -------
    dict(
      'cats' : int, float, other non array type
        The categorical values to be mapped to indices
      'cat_to_index_map' : dict
        The map from categorical values to indices.
    )

    """
    # Convert the cat_to_index_map into an index_to_cat_map, while making
    # sure it is one-to-one. Otherwise it isn't reversible.
    index_to_cat_map = {}
    for k, v in cat_to_index_map.iteritems():
      if v in index_to_cat_map:
        raise ValueError("cat_to_index_map must be one-to-one. " + str(v) + " appears twice.")
      index_to_cat_map[v] = k

    # If the missing vals isn't empty set cats to that, otherwise convert
    # the target back to a categorical value
    if len(missing_vals):
      cats = missing_vals[0]
    else:
      cats = index_to_cat_map[target]

    return {'cats': cats, 'cat_to_index_map': cat_to_index_map}


class CatToIndexNP(CatToIndex):
  """The CatToIndex class where the cats input is an numpy array. Handles any rank for 'cats'

  Attributes
  ----------
  slot_keys : list of str
    The tank's (operation's) argument keys. They define the names of the inputs to the tank.
  tube_dict : dict(
    keys - strs. The tank's (operation's) output keys. THey define the names of the outputs of the tank
    values - types. The types of the arguments outputs.
  )
    The tank's (operation's) output keys and their corresponding types.

  """

  tube_dict = {
    'target': np.ndarray,
    'missing_vals': list,
    'cat_to_index_map': dict
  }

  def _pour(self, cats, cat_to_index_map):
    """Execute the mapping in the pour (forward) direction .

    Parameters
    ----------
    cats : np.ndarray
      The categorical values to be mapped to indices
    cat_to_index_map : dict
      The map from categorical values to indices.

    Returns
    -------
    dict(
      'target': int, float, other non array type
        The result of the sum of 'a' and 'b'.
      'missing_vals': list
        The list of all the cats that were not found in cat_to_index_map.
      'cat_to_index_map' : dict
        The map from categorical values to indices.
    )

    """
    # Pull out all the cats which are not in the cat_to_index_map.
    missing_vals = cats[~np.isin(cats, cat_to_index_map.keys())].tolist()

    # Map all the categorical values to indices, setting an index of -1
    # every time an unsupported category is encoutered.
    def safe_map(cat):
      if cat in cat_to_index_map:
        return cat_to_index_map[cat]
      else:
        return -1
    target = np.vectorize(safe_map)(cats)

    return {'target': target, 'missing_vals': missing_vals, 'cat_to_index_map': cat_to_index_map}

  def _pump(self, target, missing_vals, cat_to_index_map):
    """Execute the mapping in the pump (backward) direction .

    Parameters
    ----------
    target: np.ndarray
      The result of the sum of 'a' and 'b'.
    missing_vals: list
      The list of all the cats that were not found in cat_to_index_map.
    cat_to_index_map : dict
      The map from categorical values to indices.


    Returns
    -------
    dict(
      'cats' : np.ndarray
        The categorical values to be mapped to indices
      'cat_to_index_map' : dict
        The map from categorical values to indices.
    )

    """
    missing_vals = ut.maybe_copy(missing_vals)

    # Convert the cat_to_index_map into an index_to_cat_map, while making
    # sure it is one-to-one. Otherwise it isn't reversible.
    index_to_cat_map = {}
    for k, v in cat_to_index_map.iteritems():
      if v in index_to_cat_map:
        raise ValueError("cat_to_index_map must be one-to-one. " + str(v) + " appears twice.")
      index_to_cat_map[v] = k

    # Create the function for mapping back to categorical values, filling in
    # any missing values as it goes.
    def map_back(index):
      if index != -1:
        return index_to_cat_map[index]
      else:
        cat = missing_vals.pop(0)
        return cat

    # Need to set the otypes variable for np.vectorize, otherwise it runs
    # it once to figure out the output type of map_back. This screws up
    # missing_vals.pop
    otype = str if not len(cat_to_index_map) else type(cat_to_index_map.keys()[0])
    cats = np.vectorize(map_back, otypes=[otype])(target)

    return {'cats': cats, 'cat_to_index_map': cat_to_index_map}
