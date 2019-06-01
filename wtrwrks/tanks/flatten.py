"""Flatten tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import numpy as np

class Flatten(ta.Tank):
  """The defintion of the Flatten tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'flatten'
  slot_keys = ['a']
  tube_keys = ['target', 'shape']

  def _pour(self, a):
    """Execute the Flatten tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: np.ndarray
      The array to be flattened

    Returns
    -------
    dict(
      target: dtype of 'a'
        The flattened array.
      shape: list of ints.
        The original shape of 'a'.
    )

    """
    shape = a.shape
    target = a.flatten()
    return {'target': target, 'shape': shape}

  def _pump(self, target, shape):
    """Execute the Flatten tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: dtype of 'a'
      The flattened array.
    shape: list of ints.
      The original shape of 'a'.

    Returns
    -------
    dict(
      a: np.ndarray
        The array to be flattened
    )

    """
    return {'a': target.reshape(shape)}
