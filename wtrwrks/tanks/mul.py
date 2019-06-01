"""Mul tank definition."""
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import numpy as np


class Mul(ta.Tank):
  """The defintion of the Mul tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'mul'
  slot_keys = ['a', 'b']
  tube_keys = ['target', 'smaller_size_array', 'a_is_smaller', 'missing_vals']

  def _pour(self, a, b):
    """Execute the Mul tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: np.ndarray
      The first array to be multiplied
    b: np.ndarray
      The second array to be multiplied

    Returns
    -------
    dict(
      target: np.ndarray
        The result of a*b
      smaller_size_array: np.ndarray
        Either 'a' or 'b' depending on which has fewer elements.
      a_is_smaller: bool
        Whether a is the smaller sized array.
      missing_vals: np.ndarray
        The values from either 'a' or 'b' that were lost when the other array had a zero in that location.
    )

    """
    # If a or b is not a numpy array, then cast them to it.
    if type(a) is not np.ndarray:
      a = np.array(a)
    if type(b) is not np.ndarray:
      b = np.array(b)

    # Save the array which has a fewer number of elements. Since we can
    # reconstruct the original shape of the larger array from the target.
    a_is_smaller = a.size < b.size
    if a_is_smaller:
      smaller_size_array = ut.maybe_copy(a)
    else:
      smaller_size_array = ut.maybe_copy(b)

    # Multiply them together and save all the values which were effectively
    # 'erased' by a corresponding zero in the smaller array. We don't need to
    # to it for the other array since the smaller sized array is going to be
    # saved anyway.
    target = np.array(a * b)
    if a_is_smaller:
      missing_vals = b[target == 0]
    else:
      missing_vals = a[target == 0]

    return {'target': target, 'smaller_size_array': smaller_size_array, 'a_is_smaller': a_is_smaller, 'missing_vals': missing_vals}

  def _pump(self, target, smaller_size_array, a_is_smaller, missing_vals):
    """Execute the Mul tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray
      The result of a*b
    smaller_size_array: np.ndarray
      Either 'a' or 'b' depending on which has fewer elements.
    a_is_smaller: bool
      Whether a is the smaller sized array.
    missing_vals: np.ndarray
      The values from either 'a' or 'b' that were lost when the other array had a zero in that location.

    Returns
    -------
    dict(
      a: np.ndarray
        The first array to be multiplied
      b: np.ndarray
        The second array to be multiplied
    )

    """
    # Find the value of the larger array using target and the smaller array.
    # Fill in any missing values which occured when there was a zero involved.
    if a_is_smaller:
      a = ut.maybe_copy(smaller_size_array)
      b = np.array(target / a)
      b[target == 0] = missing_vals
    else:
      a = np.array(target / smaller_size_array)
      b = ut.maybe_copy(smaller_size_array)
      a[target == 0] = missing_vals
    return {'a': a, 'b': b}
