"""Sub tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import numpy as np


class Sub(ta.Tank):
  """The defintion of the Sub tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """
  func_name = 'sub'
  slot_keys = ['a', 'b']
  tube_keys = ['target', 'smaller_size_array', 'a_is_smaller']

  def _pour(self, a, b):
    """Execute the Sub tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: np.ndarray
      The object to subtract something from.
    b: np.ndarray
      The object which substracts from something else.

    Returns
    -------
    dict(
      target: np.ndarray
        The result of a-b.
      smaller_size_array: np.ndarray
        Either 'a' or 'b' depending on which has fewer elements.
      a_is_smaller: bool
        Whether or not 'a' is the smaller size array.
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

    target = np.array(a - b)

    return {'target': target, 'smaller_size_array': smaller_size_array, 'a_is_smaller': a_is_smaller}

  def _pump(self, target, smaller_size_array, a_is_smaller):
    """Execute the Sub tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray
      The result of a-b.
    smaller_size_array: np.ndarray
      Either 'a' or 'b' depending on which has fewer elements.
    a_is_smaller: bool
      Whether or not 'a' is the smaller size array.

    Returns
    -------
    dict(
      a: np.ndarray
        The object to subtract something from.
      b: np.ndarray
        The object which substracts from something else.
    )

    """
    # Reconstruct the larger array from the smaller size array nd the target.
    if a_is_smaller:
      a = ut.maybe_copy(smaller_size_array)
      b = np.array(a - target)
    else:
      a = np.array(target + smaller_size_array)
      b = ut.maybe_copy(smaller_size_array)

    return {'a': a, 'b': b}
