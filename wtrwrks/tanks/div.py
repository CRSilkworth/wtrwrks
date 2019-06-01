"""Div tank definition."""
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import numpy as np


class Div(ta.Tank):
  """The defintion of the Div tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """
  func_name = 'div'
  slot_keys = ['a', 'b']
  tube_keys = ['target', 'smaller_size_array', 'a_is_smaller', 'missing_vals', 'remainder']

  def _pour(self, a, b):
    """Execute the Div tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: np.ndarray
      The numerator array.
    b: np.ndarray
      The denominator array

    Returns
    -------
    dict(
      target: np.ndarray
        The result of a/b
      smaller_size_array: np.ndarray
        Either 'a' or 'b' depending on which has fewer elements.
      a_is_smaller: bool
        Whether a is the smaller sized array.
      missing_vals: np.ndarray
        The values from either 'a' or 'b' that were lost when the other array had a zero in that location.
      remainder: np.ndarray
        The remainder of a/b in the case that 'a' and 'b' are of integer type.
    )

    """
    # If they aren't numpy arrays then cast them to arrays.
    if type(a) is not np.ndarray:
      a = np.array(a)
    if type(b) is not np.ndarray:
      b = np.array(b)

    # Find the array with fewer elements and save that.
    a_is_smaller = a.size < b.size
    if a_is_smaller:
      smaller_size_array = ut.maybe_copy(a)
    else:
      smaller_size_array = ut.maybe_copy(b)

    # Do the division
    target = np.array(a / b)

    # Save the values of the larger array whose values are erased by a zero in
    # the smaller array
    if a_is_smaller:
      missing_vals = b[(target == 0)]
    else:
      missing_vals = a[np.isposinf(target) | np.isneginf(target) | np.isnan(target)]

    # Don't allowed integer division by zero.
    if a.dtype in (np.int32, np.int64) and b.dtype in (np.int32, np.int64):
      if (b == 0).any():
        raise ZeroDivisionError("Integer division by zero is not supported.")
      remainder = np.array(np.remainder(a, b))
    else:
      remainder = np.array([], dtype=target.dtype)

    return {'target': target, 'smaller_size_array': smaller_size_array, 'a_is_smaller': a_is_smaller, 'missing_vals': missing_vals, 'remainder': remainder}

  def _pump(self, target, smaller_size_array, a_is_smaller, missing_vals, remainder):
    """Execute the Div tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray
      The result of a/b
    smaller_size_array: np.ndarray
      Either 'a' or 'b' depending on which has fewer elements.
    a_is_smaller: bool
      Whether a is the smaller sized array.
    missing_vals: np.ndarray
      The values from either 'a' or 'b' that were lost when the other array had a zero in that location.
    remainder: np.ndarray
      The remainder of a/b in the case that 'a' and 'b' are of integer type.

    Returns
    -------
    dict(
      a: np.ndarray
        The numerator array.
      b: np.ndarray
        The denominator array
    )

    """
    if a_is_smaller:
      # If a is the smaller of the two arrays, then it was the one that was
      # saved. So no need to worry about the remainder.
      a = ut.maybe_copy(smaller_size_array)
      b = np.array(a / target)
      b[(target == 0)] = missing_vals
    else:
      a = target * smaller_size_array
      if target.dtype in (np.int32, np.int64):
        a = np.array(a + remainder)
      b = ut.maybe_copy(smaller_size_array)

      # If b is the smaller array then it is the one that was saved. This means
      # a nan, negative infinity, or positive infinity, (i.e. zeros in b)
      # correspond to the missing values in a.
      a[np.isposinf(target) | np.isneginf(target) | np.isnan(target)] = missing_vals
    return {'a': a, 'b': b}
