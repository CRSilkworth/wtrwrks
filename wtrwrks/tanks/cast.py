"""Cast tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import numpy as np


class Cast(ta.Tank):
  """The defintion of the Cast tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'cast'
  slot_keys = ['a', 'dtype']
  tube_keys = ['target', 'input_dtype', 'diff']

  def _pour(self, a, dtype):
    """Execute the Cast tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: np.ndarray
      The object to be casted to a new type. Must be able to be converted into a numpy array.
    dtype: a numpy dtype
      The type to cast the object to.

    Returns
    -------
    dict(
      target: The type specified by the dtype slot
        The result of casting 'a' to the new dtype.
      input_dtype: a numpy dtype
        The dtype of the original array.
      diff: The datatype of the original 'a' array
        The difference between the original 'a' and the casted array.
    )

    """
    a = np.array(a)
    target = a.astype(dtype)

    # If the inputted is of float type and is being cast to an integer, it will
    # lose information. So the difference between the original and casted array
    # must be saved.
    if a.dtype in (np.float64, np.float32) and dtype in (np.int32, np.int64, np.bool, int):
      diff = a - target
    else:
      diff = np.zeros_like(target)
    return {'target': target, 'diff': diff, 'input_dtype': a.dtype}

  def _pump(self, target, input_dtype, diff):
    """Execute the Cast tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: The type specified by the dtype slot
      The result of casting 'a' to the new dtype.
    input_dtype: a numpy dtype
      The dtype of the original array.
    diff: The datatype of the original 'a' array
      The difference between the original 'a' and the casted array.

    Returns
    -------
    dict(
      a: np.ndarray
        The object to be casted to a new type. Must be able to be converted into a numpy array.
      dtype: a numpy dtype
        The type to cast the object to.
    )

    """
    dtype = target.dtype

    a = target
    if np.issubdtype(input_dtype, np.number):
      a = diff + target
    a = a.astype(input_dtype)
    return {'a': a, 'dtype': dtype}
