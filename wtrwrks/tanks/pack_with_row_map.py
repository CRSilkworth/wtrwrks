"""Concatenate tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import numpy as np
import wtrwrks.tanks.utils as ut
import logging


class PackWithRowMap(ta.Tank):
  """More efficiently pack in the data of an array by overwriting the default_val's. The array must have rank at least equal to 2 The last dimension will be packed so that fewer default vals appear, and the next to last dimension with be shortened, any other dimensions are left unchanged.
  e.g.

  default_val = 0
  a = np.array([
    [1, 1, 1, 0, 0, 0],
    [2, 2, 0, 0, 0, 0],
    [3, 3, 0, 3, 3, 0],
    [4, 0, 0, 0, 0, 0],
    [5, 5, 5, 5, 5, 5],
    [6, 6, 0, 0, 0, 0],
    [7, 7, 0, 0, 0, 0]
  ])

  target = np.array([
    [1, 1, 1, 2, 2, 0],
    [3, 3, 3, 3, 4, 0],
    [5, 5, 5, 5, 5, 5],
    [6, 6, 7, 7, 0, 0]
  ])

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'pack_with_row_map'
  slot_keys = ['a', 'default_val', 'row_map']
  tube_keys = ['target', 'row_map', 'default_val']
  pass_through_keys = ['default_val', 'row_map']

  def _pour(self, a, row_map, default_val):
    """

    Parameters
    ----------
    a: np.ndarray
      The array to pack
    default_val: np.ndarray.dtype
      The value that will be allowed to be overwritten in the packing process.
    row_map: np.ndarray
      A mapping from the new rows to the original rows.

    Returns
    -------
    dict(
      target: np.ndarray
        The packed version of the 'a' array. Has same dims except for the second to last dimension which is usually shorter.
      default_val: np.ndarray.dtype
        The value that will be allowed to be overwritten in the packing process.
      row_map: np.ndarray
        A mapping from the new rows to the original rows.
    )

    """
    logging.debug('%s', a.shape)
    a = np.array(a)

    a_row_dim = a.shape[-1]
    sliced_row_map = row_map[..., 0]

    target_inner_dims = list(sliced_row_map.shape[-2:])
    reshaped_a = a.reshape([-1, a_row_dim])
    reshaped_row_map = sliced_row_map.reshape([-1] + target_inner_dims)

    # Flatten any of the outer dimensions and work with two d arrays, since
    # those will be the dimensions which affect the packing.
    target = []
    for slice_a, slice_row_map, in zip(reshaped_a, reshaped_row_map):
      slice_target = slice_a[slice_row_map]
      slice_target[slice_row_map == -1] = default_val
      target.append(slice_target.tolist())

    target = np.stack(target)
    target = target.reshape(row_map.shape[:-1])

    return {'target': target, 'default_val': default_val, 'row_map': row_map}

  def _pump(self, target, default_val, row_map):
    """Execute the Concatenate tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray
      The packed version of the 'a' array. Has same dims except for the second to last dimension which is usually shorter.
    default_val: np.ndarray.dtype
      The value that will be allowed to be overwritten in the packing process.
    row_map: np.ndarray
      A mapping from the new rows to the original rows.

    Returns
    -------
    dict(
      a: np.ndarray
        The array to pack
      default_val: np.ndarray.dtype
        The value that will be allowed to be overwritten in the packing process.
      row_map: np.ndarray
        A mapping from the new rows to the original rows.
    )

    """
    target = np.array(target)
    dtype = target.dtype
    row_dim = target.shape[-2]
    col_dim = target.shape[-1]

    a_row_dim = np.max(row_map[..., 0]) + 1

    reshaped_target = target.reshape([-1, row_dim, col_dim])

    sliced_row_map = row_map[..., 0]
    reshaped_row_map = sliced_row_map.reshape([-1, row_dim, col_dim])

    a = []
    for slice_target, slice_row_map, in zip(reshaped_target, reshaped_row_map):
      slice_a = np.full([a_row_dim], default_val, dtype=dtype).tolist()

      for target_row_num in xrange(slice_target.shape[0]):
        for col_num, a_row_num in enumerate(slice_row_map[target_row_num]):
          if a_row_num == -1:
            continue

          slice_a[a_row_num] = slice_target[target_row_num][col_num]
      a.append(slice_a)
    a = np.stack(a).reshape(list(target.shape[:-2]) + [a_row_dim])
    return {'a': a, 'default_val': default_val, 'row_map': row_map}
