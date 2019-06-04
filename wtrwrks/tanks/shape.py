"""Shape tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import numpy as np
import wtrwrks.tanks.utils as ut

class Shape(ta.Tank):
  """Get the shape of an array.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'shape'
  slot_keys = ['a']
  tube_keys = ['target', 'a']
  pass_through_keys = ['a']

  def _pour(self, a):
    """

    Parameters
    ----------
    a: np.ndarray
      The array to get the shape of

    Returns
    -------
    dict(
      target: list of ints
        The shape of the array.
      a: np.ndarray
        The array to get the shape of
    )

    """
    return {'target': list(a.shape), 'a': ut.maybe_copy(a)}

  def _pump(self, target, a):
    """Execute the Shape tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: list of ints
      The shape of the array.
    a: np.ndarray
      The array to get the shape of

    Returns
    -------
    dict(
      a: np.ndarray
        The array to get the shape of
    )

    """
    return {'a': ut.maybe_copy(a)}
