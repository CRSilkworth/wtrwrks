"""Concatenate tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import numpy as np


class Concatenate(ta.Tank):
  """The defintion of the Concatenate tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'concatenate'
  slot_keys = ['a_list', 'axis']
  tube_keys = ['target', 'axis', 'indices', 'dtypes']
  pass_through_keys = ['axis']

  def _pour(self, a_list, axis):
    """Execute the Concatenate tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a_list: list of arrays
      The list of arrays to concatenate.
    axis: int
      The axis along which to concatenate.

    Returns
    -------
    dict(
      target: np.ndarray
        The concatenation of 'a_list' along 'axis'
      axis: int
        The axis along which to concatenate.
      indices: np.ndarray
        The indices that mark the separation of arrays.
      dtypes: list of dtypes
        The dtypes of the original elements of 'a_list'. Must be of the same length as the orignal 'a_list.'
    )

    """
    indices = []
    dtypes = []
    # Keep a cursor going which represents the current index in the outputted
    # so that it can be split in the right place during pump.
    cursor = 0
    for a_num, a in enumerate(a_list):
      a = np.array(a)

      # Save the dtypes since this can be lots when the concatenation is of
      # separate types.
      dtypes.append(a.dtype)

      # If this is the last item the continue since otherwise it'll create one
      # split too many
      if a_num == len(a_list) - 1:
        continue

      # Append the current value of the cursor plus the size of the dimension.
      indices.append(cursor + a.shape[axis])
      cursor += a.shape[axis]
    return {'target': np.concatenate(a_list, axis=axis), 'indices': np.array(indices), 'axis': axis, 'dtypes': dtypes}

  def _pump(self, target, indices, axis, dtypes):
    """Execute the Concatenate tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray
      The concatenation of 'a_list' along 'axis'
    axis: int
      The axis along which to concatenate.
    indices: np.ndarray
      The indices that mark the separation of arrays.
    dtypes: list of dtypes
      The dtypes of the original elements of 'a_list'. Must be of the same length as the orignal 'a_list.'

    Returns
    -------
    dict(
      a_list: list of arrays
        The list of arrays to concatenate.
      axis: int
        The axis along which to concatenate.
    )

    """
    # Split up the array according the the indices saved from the pour.
    splits = np.split(target, indices, axis=axis)
    a_list = []
    for a, dtype in zip(splits, dtypes):

      # caste them all back to their original types.
      a = a.astype(dtype)
      a_list.append(a)
    return {'a_list': a_list, 'axis': axis}
