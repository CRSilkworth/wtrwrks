"""Reshape tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import numpy as np


class Remove(ta.Tank):
  """Remove elements of an array according to a mask.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'remove'
  slot_keys = ['a', 'mask']
  tube_keys = ['target', 'mask', 'removed']
  pass_through_keys = ['mask']

  def _pour(self, a, mask):
    """Execute the Flatten tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: np.ndarray
      The array to execute the remove on
    mask: np.ndarray
      An array of Trues and Falses, telling which elements to remove. Either needs to be the same shape as a or broadcastable to a.

    Returns
    -------
    dict(
      target: np.ndarray
        The array with elements removed
      removed: np.ndarray
        The remove elements of the array
      mask: np.ndarray
        An array of Trues and Falses, telling which elements to remove. Either needs to be the same shape as a or broadcastable to a.
    )

    """
    print a.shape, mask.shape
    target = a[mask]
    removed = a[~mask]
    return {'target': target, 'mask': mask, 'removed': removed}

  def _pump(self, target, mask, removed):
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
      target: np.ndarray
        The array with elements removed
      removed: np.ndarray
        The remove elements of the array
      mask: np.ndarray
        An array of Trues and Falses, telling which elements to remove. Either needs to be the same shape as a or broadcastable to a.
    )

    """
    if target.dtype.type in (np.unicode_, np.string_):
      if target.dtype.itemsize > removed.dtype.itemsize:
        dtype = target.dtype.itemsize
      else:
        dtype = removed.dtype.itemsize
    else:
      dtype = target.dtype

    shape = list(mask.shape)
    shape = shape + list(target.shape[len(shape):])
    a = np.empty(shape, dtype=dtype)

    a[mask] = target
    a[~mask] = removed
    return {'a': a, 'mask': mask}
