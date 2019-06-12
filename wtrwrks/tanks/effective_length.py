"""EffectiveLength tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import numpy as np
import wtrwrks.tanks.utils as ut

class EffectiveLength(ta.Tank):
  """Get the length of the last dimension, not including the default_val.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'effective_length'
  slot_keys = ['a', 'default_val']
  tube_keys = ['target', 'a', 'default_val']
  pass_through_keys = ['a', 'default_val']

  def _pour(self, a, default_val):
    """

    Parameters
    ----------
    a: np.ndarray
      The array to get the effective length of.
    default_val:
      The value to not count

    Returns
    -------
    dict(
      target: np.ndarray
        An array of the same shape as 'a' except missing the last dimension. The values are effective lengths of the last dimesion of a.
      a: np.ndarray
        The array to get the effective length of.
      default_val:
        The value to not count
    )

    """
    zero = (np.array(a) == default_val)

    all_zero = np.all(zero, axis=-1)
    not_zero = ~zero

    reversed_last_dim = not_zero[..., ::-1]

    lengths = np.argmax(reversed_last_dim, axis=-1)
    lengths = a.shape[-1] - lengths
    lengths[all_zero] = 0

    return {'target': lengths, 'a': ut.maybe_copy(a), 'default_val': default_val}

  def _pump(self, target, a, default_val):
    """Execute the Shape tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray
      An array of the same shape as 'a' except missing the last dimension. The values are effective lengths of the last dimesion of a.
    a: np.ndarray
      The array to get the effective length of.
    default_val:
      The value to not count

    Returns
    -------
    dict(
      a: np.ndarray
        The array to get the effective length of.
      default_val:
        The value to not count
    )

    """

    return {'a': ut.maybe_copy(a), 'default_val': default_val}
