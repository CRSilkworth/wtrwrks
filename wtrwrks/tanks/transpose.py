"""Transpose tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import numpy as np


class Transpose(ta.Tank):
  """The defintion of the Transpose tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

    Attributes
    ----------
    slot_keys: list of strs
      The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
    tubes: list of strs
      The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'transpose'
  slot_keys = ['a', 'axes']
  tube_keys = ['target', 'axes']
  pass_through_keys = ['axes']

  def _pour(self, a, axes):
    """Execute the Transpose tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: np.ndarray
      The array to be transposed.
    axes: list of ints
      The permutation of axes. len(axes) must equal rank of 'a', and each integer from 0 to len(axes) - 1 must appear exactly once.

    Returns
    -------
    dict(
      target: np.ndarray
        The transposed array.
      axes: list of ints
        The permutation of axes. len(axes) must equal rank of 'a', and each integer from 0 to len(axes) - 1 must appear exactly once.
    )

    """
    return {'target': np.transpose(a, axes=axes), 'axes': axes}

  def _pump(self, target, axes):
    """Execute the Transpose tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray
      The transposed array.
    axes: list of ints
      The permutation of axes. len(axes) must equal rank of 'a', and each integer from 0 to len(axes) - 1 must appear exactly once.

    Returns
    -------
    dict(
      a: np.ndarray
        The array to be transposed.
      axes: list of ints
        The permutation of axes. len(axes) must equal rank of 'a', and each integer from 0 to len(axes) - 1 must appear exactly once.
    )

    """
    # The inverse of a permutation is simply the argsort of that permutation.
    trans_axes = np.argsort(axes)
    a = np.transpose(target, axes=trans_axes)
    return {'a': a, 'axes': axes}
