"""Reshape tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import numpy as np


class Tile(ta.Tank):
  """Tile the elements of an array into an array with a shape defined by reps.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'tile'
  slot_keys = ['a', 'reps']
  tube_keys = ['target', 'old_shape', 'reps']
  pass_through_keys = ['reps']

  def _pour(self, a, reps):
    """Execute the Tile tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: np.ndarray
      The array to reshape
    reps : list of ints
      The number of times to tile the array in each dimension

    Returns
    -------
    dict(
      target: np.ndarray
        The reshaped array
      old_shape: list of ints
        The old shape of the array
      reps : list of ints
        The number of times to tile the array in each dimension
    )

    """
    a = np.array(a)
    old_shape = a.shape
    target = np.tile(a, reps)

    return {'target': target, 'old_shape': old_shape, 'reps': reps}

  def _pump(self, target, old_shape, reps):
    """Execute the Tile tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray
      The reshaped array
    old_shape: list of ints
      The old shape of the array
    reps : list of ints
      The number of times to tile the array in each dimension

    Returns
    -------
    dict(
      a: np.ndarray
        The array to reshape
      reps : list of ints
        The number of times to tile the array in each dimension
    )

    """
    shape = target.shape
    slice_indices = [0] * (len(shape) - len(old_shape))

    for dim in old_shape:
      slice_indices.append(slice(0, dim))

    a = target[tuple(slice_indices)]

    return {'a': a, 'reps': reps}
