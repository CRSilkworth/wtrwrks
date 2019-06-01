"""Add tank definition."""
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import numpy as np


class Add(ta.Tank):
  """The defintion of the CatToIndex tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """
  func_name = 'add'
  slot_keys = ['a', 'b']
  tube_keys = ['target', 'smaller_size_array', 'a_is_smaller']

  def _pour(self, a, b):
    """Execute the CatToIndex tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    cats: np.ndarray
      The array with all the category values to map to indices.
    cat_to_index_map: dict
      The mapping from category value to index. Must be one to one and contain all indices from zero to len(cat_to_index_map) - 1

    Returns
    -------
    dict(
      target: np.ndarray of ints
        The indices of all the corresponding category values from 'cats'.
      cat_to_index_map: dict
        The mapping from category value to index. Must be one to one and contain all indices from zero to len(cat_to_index_map) - 1
      missing_vals: list of category values
        All the category values from 'cats' which were not found in cat_to_index_map.
      input_dtype: a numpy dtype
        The dtype of the inputted 'cats' array.
    )

    """

    # Convert to nump arrays
    if type(a) is not np.ndarray:
      a = np.array(a)
    if type(b) is not np.ndarray:
      b = np.array(b)

    # Copy whichever has a fewer number of elements and pass as output
    a_is_smaller = a.size < b.size
    if a_is_smaller:
      smaller_size_array = ut.maybe_copy(a)
    else:
      smaller_size_array = ut.maybe_copy(b)

    target = np.array(a + b)

    return {'target': target, 'smaller_size_array': smaller_size_array, 'a_is_smaller': a_is_smaller}

  def _pump(self, target, smaller_size_array, a_is_smaller):
    """Execute the CatToIndex tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray of ints
      The indices of all the corresponding category values from 'cats'.
    cat_to_index_map: dict
      The mapping from category value to index. Must be one to one and contain all indices from zero to len(cat_to_index_map) - 1
    missing_vals: list of category values
      All the category values from 'cats' which were not found in cat_to_index_map.
    input_dtype: a numpy dtype
      The dtype of the inputted 'cats' array.

    Returns
    -------
    dict(
      cats: np.ndarray
        The array with all the category values to map to indices.
      cat_to_index_map: dict
        The mapping from category value to index. Must be one to one and contain all indices from zero to len(cat_to_index_map) - 1
    )

    """
    # reconstruct the other array from the smaller size array nd the target.
    if a_is_smaller:
      a = ut.maybe_copy(smaller_size_array)
      b = np.array(target - a)
    else:
      a = np.array(target - smaller_size_array)
      b = ut.maybe_copy(smaller_size_array)

    return {'a': a, 'b': b}
