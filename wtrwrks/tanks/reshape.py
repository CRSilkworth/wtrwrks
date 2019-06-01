"""Reshape tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import numpy as np


class Reshape(ta.Tank):
  """Reshape an array.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'reshape'
  slot_keys = ['a', 'shape']
  tube_keys = ['target', 'old_shape']

  def _pour(self, a, shape):
    """Execute the Flatten tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: np.ndarray
      The array to reshape
    shape : list of ints
      The new shape of the array.

    Returns
    -------
    dict(
      target: np.ndarray
        The reshaped array
      old_shape: list of ints
        The old shape of the array
    )

    """
    old_shape = a.shape
    target = a.reshape(shape)
    return {'target': target, 'old_shape': old_shape}

  def _pump(self, target, old_shape):
    """Execute the Flatten tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray
      The reshaped array
    old_shape: list of ints
      The old shape of the array

    Returns
    -------
    dict(
      a: np.ndarray
        The array to reshape
      shape : list of ints
        The new shape of the array.
    )

    """
    shape = target.shape
    return {'a': target.reshape(old_shape), 'shape': shape}
