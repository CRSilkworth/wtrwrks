"""Split tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import numpy as np


class Split(ta.Tank):
  """The defintion of the Split tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'split'
  slot_keys = ['a', 'indices', 'axis']
  tube_keys = ['target', 'indices', 'axis']
  pass_through_keys = ['indices', 'axis']

  def _pour(self, a, indices, axis):
    """Execute the Split tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: np.ndarray
      The array to split up.
    indices: np.ndarray
      The indices of the points to split up the array.
    axis: int
      The axis along which to split up the array.

    Returns
    -------
    dict(
      target: list of arrays
        The list of split up arrays.
      indices: np.ndarray
        The indices of the points to split up the array.
      axis: int
        The axis along which to split up the array.
    )

    """
    return {'target': np.split(a, indices, axis=axis), 'indices': indices, 'axis': axis}

  def _pump(self, target, indices, axis):
    """Execute the Split tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: list of arrays
      The list of split up arrays.
    indices: np.ndarray
      The indices of the points to split up the array.
    axis: int
      The axis along which to split up the array.

    Returns
    -------
    dict(
      a: np.ndarray
        The array to split up.
      indices: np.ndarray
        The indices of the points to split up the array.
      axis: int
        The axis along which to split up the array.
    )

    """
    a = np.concatenate(target, axis=axis)
    return {'a': a, 'indices': indices, 'axis': axis}
