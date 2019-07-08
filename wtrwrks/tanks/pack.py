"""Concatenate tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import numpy as np
import wtrwrks.tanks.utils as ut
import logging

class Pack(ta.Tank):
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

  func_name = 'pack'
  slot_keys = ['a', 'lengths', 'default_val', 'max_group']
  tube_keys = ['target', 'ends', 'row_map', 'default_val', 'max_group']
  pass_through_keys = ['default_val', 'max_group']

  def _pour(self, a, lengths, default_val, max_group):
    """

    Parameters
    ----------
    a: np.ndarray
      The array to pack
    lengths: np.ndarray
      The of lengths of 'valid' data. The not valid data will be overwritten when it's packed together.
    max_group: int
      Maximum number of original rows of data packed into a single row.
    default_val: np.ndarray.dtype
      The value that will be allowed to be overwritten in the packing process.

    Returns
    -------
    dict(
      target: np.ndarray
        The packed version of the 'a' array. Has same dims except for the second to last dimension which is usually shorter.
      ends: np.ndarray
        The endpoints of all the original rows within the packed array.
      row_map: np.ndarray
        A mapping from the new rows to the original rows.
      default_val: np.ndarray.dtype
        The value that will be allowed to be overwritten in the packing process.
      max_group: int
        Maximum number of original rows of data packed into a single row.
    )

    """
    logging.debug('%s', a.shape)
    a = np.array(a)
    dtype = a.dtype
    outer_dims = list(a.shape[:-2])
    row_dim = a.shape[-2]
    col_dim = a.shape[-1]

    pack_row_lens = []
    grouped_row_lens = []
    all_pack_rows = []
    all_ends = []
    reshaped_a = a.reshape([-1, row_dim, col_dim])
    reshaped_lengths = lengths.reshape([-1, row_dim])
    row_map = []

    # Flatten any of the outer dimensions and work with two d arrays, since
    # those will be the dimensions which affect the packing.
    for slice_num, two_d_slice in enumerate(reshaped_a):
      cur_lengths = reshaped_lengths[slice_num]

      tally = 0
      grouped_rows = []
      rows = []
      triplets = []
      slice_ends = []
      slice_row_map = []

      ends = np.zeros([col_dim], dtype=bool)
      # Assign each row from the two d slice to a group of rows adding rows
      # until the total number of elements in that row is reached. Then start
      # a new group of rows.
      for row_num, num in enumerate(cur_lengths):
        if num + tally > col_dim or len(rows) == max_group:
          grouped_rows.append(rows)
          grouped_row_lens.append(len(rows))
          slice_row_map.append(triplets)

          slice_ends.append(ends)
          ends = np.zeros([col_dim], dtype=bool)
          ends[num - 1] = True
          triplets = []
          rows = []
          tally = 0
        elif num:
          ends[num + tally - 1] = True
        triplets.append([row_num, tally, tally + num])
        rows.append(row_num)
        tally += num

      slice_row_map.append(triplets)
      grouped_row_lens.append(len(rows))
      grouped_rows.append(rows)
      slice_ends.append(ends)

      # Create the new rows of the arrays by pulling out the non default values
      # from the two d slice and adding them to the new row.
      pack_rows = []
      for new_row_num, old_rows in enumerate(grouped_rows):
        pack_row = []
        for old_row in old_rows:
          # Pull out the non default values from this old row, add to pack_row
          pack_row.extend(
            two_d_slice[old_row][:cur_lengths[old_row]].tolist()
          )

        # Standardize the length of pack row by adding in default vals
        pack_row = pack_row + [default_val] * (col_dim - len(pack_row))
        pack_rows.append(np.array(pack_row, dtype=dtype))

      row_map.append(slice_row_map)
      pack_rows = np.stack(pack_rows)
      pack_row_lens.append(pack_rows.shape[0])
      all_pack_rows.append(pack_rows)
      all_ends.append(slice_ends)
    max_num_rows = max(pack_row_lens)

    # Standardize the size of each newly created slice (pack_rows) so that they
    # can all be added to one array. pack_rows slice is filled with several
    # rows of all default_vals if it does not have the max_num_rows of filled
    # Rows.
    target = []
    for slice_num, (pack_row_len, pack_rows) in enumerate(zip(pack_row_lens, all_pack_rows)):
      rows_to_add = max_num_rows - pack_row_len
      default_val_array = np.full([rows_to_add, col_dim], default_val, dtype=dtype)
      target.append(np.concatenate([pack_rows, default_val_array]))

      falses = np.zeros([rows_to_add, col_dim], dtype=bool)
      all_ends[slice_num] = np.concatenate([all_ends[slice_num], falses])

      for row_num in xrange(len(row_map[slice_num])):
        row_map[slice_num][row_num] = row_map[slice_num][row_num] + [[-1, -1, -1]] * (max_group - len(row_map[slice_num][row_num]))

      row_map[slice_num] = np.stack(row_map[slice_num])
      minus_ones = -1 * np.ones([rows_to_add, max_group, 3], dtype=int)
      row_map[slice_num] = np.concatenate([row_map[slice_num], minus_ones])

    # Reshape the array so that you have an array with only the second to last
    # dimension differening in size from the original 'a'
    target = np.stack(target)
    target = target.reshape(outer_dims + [max_num_rows, col_dim])

    row_map = np.stack(row_map).reshape(outer_dims + [max_num_rows, max_group, 3])

    all_ends = np.stack(all_ends).reshape(outer_dims + [max_num_rows, col_dim])

    return {'target': target, 'default_val': default_val, 'ends': all_ends, 'row_map': row_map, 'max_group': max_group}

  def _pump(self, target, default_val, row_map, ends, max_group):
    """Execute the Concatenate tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray
      The packed version of the 'a' array. Has same dims except for the second to last dimension which is usually shorter.
    ends: np.ndarray
      The endpoints of all the original rows within the packed array.
    row_map: np.ndarray
      A mapping from the new rows to the original rows.
    default_val: np.ndarray.dtype
      The value that will be allowed to be overwritten in the packing process.
    max_group: int
      Maximum number of original rows of data packed into a single row.

    Returns
    -------
    dict(
      a : np.ndarray
        The array to pack
      lengths: np.ndarray
        The of lengths of 'valid' data. The not valid data will be overwritten when it's packed together.
      max_group: int
        Maximum number of original rows of data packed into a single row.
      default_val: np.ndarray.dtype
        The value that will be allowed to be overwritten in the packing process.
    )

    """
    target = np.array(target)
    dtype = target.dtype
    row_dim = target.shape[-2]
    col_dim = target.shape[-1]

    a_row_dim = np.max(row_map[..., 0]) + 1

    reshaped_target = target.reshape([-1, row_dim, col_dim])
    reshaped_row_map = row_map.reshape([-1, row_dim] + list(row_map.shape[-2:]))

    a = []
    lengths = []

    for two_d_slice, cur_row_map in zip(reshaped_target, reshaped_row_map):
      recon = np.full([a_row_dim, col_dim], default_val, dtype=dtype)
      slice_lengths = np.zeros([a_row_dim], dtype=int)

      for packed_row_num in xrange(row_dim):
        packed_row = two_d_slice[packed_row_num]
        for row_num, b_index, e_index in cur_row_map[packed_row_num]:
          if row_num == -1:
            break

          length = e_index - b_index
          recon[row_num][:length] = packed_row[b_index: e_index]
          slice_lengths[row_num] = length

      lengths.append(slice_lengths)
      recon = np.array(recon, dtype=dtype)
      a.append(recon)

    lengths = np.stack(lengths)
    lengths = lengths.reshape(list(target.shape[:-2]) + [a_row_dim])
    a = np.stack(a)
    a = a.reshape(list(target.shape[:-2]) + [a_row_dim, col_dim])

    return {'a': a, 'default_val': default_val, 'lengths': lengths, 'max_group': max_group}
