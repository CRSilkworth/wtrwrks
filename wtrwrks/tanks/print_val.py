"""Print tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import numpy as np
import wtrwrks.tanks.utils as ut

class Print(ta.Tank):
  """Get the shape of an array.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'print_val'
  slot_keys = None
  tube_keys = None
  slot_and_tube_equal = None

  def _pour(self, **kwargs):
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
    for key in kwargs:
      print kwargs[key]
    return kwargs

  def _pump(self, **kwargs):
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
    for key in kwargs:
      print kwargs[key]
    return kwargs