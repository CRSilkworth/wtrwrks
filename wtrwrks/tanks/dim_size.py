"""Shape tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import numpy as np
import wtrwrks.tanks.utils as ut


class DimSize(ta.Tank):
  """Get the size of a dimension of an array.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'dim_size'
  slot_keys = ['a', 'axis']
  tube_keys = ['target', 'a', 'axis']
  pass_through_keys = ['a', 'axis']

  def _pour(self, a, axis):
    """

    Parameters
    ----------
    a: np.ndarray
      The array to get the shape of
    axis: int
      The axis to get the dim_size from.

    Returns
    -------
    dict(
      target: list of ints
        The shape of the array.
      a: np.ndarray
        The array to get the shape of
      axis: int
        The axis to get the dim_size from.
    )

    """
    return {'target': a.shape[axis], 'a': ut.maybe_copy(a), 'axis': axis}

  def _pump(self, target, a, axis):
    """Execute the Shape tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: list of ints
      The shape of the array.
    a: np.ndarray
      The array to get the shape of
    axis: int
      The axis to get the dim_size from.

    Returns
    -------
    dict(
      a: np.ndarray
        The array to get the shape of
      axis: int
        The axis to get the dim_size from.
    )

    """
    return {'a': ut.maybe_copy(a), 'axis': axis}
