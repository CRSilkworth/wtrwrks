"""Concatenate tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import numpy as np


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
    [0, 6, 6, 0, 0, 0],
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

  slot_keys = ['a', 'default_val']
  tube_keys = ['target', 'default_val', 'is_default', 'ends']

  def _pour(self, a, default_val):
    """

    Parameters
    ----------
    a: np.ndarray
      The array to pack
    default_val: np.ndarray.dtype
      The value that will be allowed to be overwritten in the packing process.

    Returns
    -------
    dict(
      target: np.ndarray
        The packed version of the 'a' array. Has same dims except for the second to last dimension which is usually shorter.
      default_val: np.ndarray.dtype
        The value that will be allowed to be overwritten in the packing process.
      is_default: np.ndarray of bools
        An array which specifies which elements of the original 'a' have a value equal to 'defaul_val'
    )

    """
    a = np.array(a)
    dtype = a.dtype
    outer_dims = list(a.shape[:-2])
    row_dim = a.shape[-2]
    col_dim = a.shape[-1]

    pack_row_lens = []
    all_pack_rows = []
    all_ends = []
    is_default = a == default_val
    reshaped_is_default = is_default.reshape([-1, row_dim, col_dim])

    # Flatten any of the outer dimensions and work with two d arrays, since
    # those will be the dimensions which affect the packing.
    for slice_num, two_d_slice in enumerate(a.reshape([-1, row_dim, col_dim])):
      cur_is_default = reshaped_is_default[slice_num]

      # Find the number of non default vals on each row
      num_non_defaults = np.sum(~cur_is_default, axis=-1)

      grouped_rows = []
      tally = 0
      rows = []
      slice_ends = []
      ends = np.zeros([col_dim], dtype=bool)
      # Assign each row from the two d slice to a group of rows adding rows
      # until the total number of elements in that row is reached. Then start
      # a new group of rows.
      for row_num, num in enumerate(num_non_defaults):
        if num + tally > col_dim:
          grouped_rows.append(rows)
          slice_ends.append(ends)
          ends = np.zeros([col_dim], dtype=bool)
          ends[num - 1] = True
          rows = []
          tally = 0
        elif num:
          ends[num + tally - 1] = True

        rows.append(row_num)
        tally += num

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
            two_d_slice[old_row][~cur_is_default[old_row]].tolist()
          )

        # Standardize the length of pack row by adding in default vals
        pack_row = pack_row + [default_val] * (col_dim - len(pack_row))
        pack_rows.append(np.array(pack_row, dtype=dtype))

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

    # Reshape the array so that you have an array with only the second to last
    # dimension differening in size from the original 'a'
    target = np.stack(target)
    target = target.reshape(outer_dims + [max_num_rows, col_dim])

    all_ends = np.stack(all_ends).reshape(outer_dims + [max_num_rows, col_dim])

    return {'target': target, 'is_default': is_default, 'default_val': default_val, 'ends': all_ends}

  def _pump(self, target, is_default, default_val, ends):
    """Execute the Concatenate tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray
      The packed version of the 'a' array. Has same dims except for the second to last dimension which is usually shorter.
    default_val: np.ndarray.dtype
      The value that will be allowed to be overwritten in the packing process.
    is_default: np.ndarray of bools
      An array which specifies which elements of the original 'a' have a value equal to 'defaul_val'

    Returns
    -------
    dict(
      a: np.ndarray
        The array to pack
      default_val: np.ndarray.dtype
        The value that will be allowed to be overwritten in the packing process.
    )

    """
    target = np.array(target)
    dtype = target.dtype
    row_dim = target.shape[-2]
    col_dim = target.shape[-1]

    reshaped_target = target.reshape([-1, row_dim, col_dim])
    reshaped_is_default = is_default.reshape([-1] + list(is_default.shape[-2:]))

    a = []
    for two_d_slice, mask in zip(reshaped_target, reshaped_is_default):
      recon = []

      # Remove any left over default_values
      packed = two_d_slice[two_d_slice != default_val].flatten().tolist()

      # Go along each value in the is_default, fill with either a default val
      # or the next non default val depending on whether the original a had
      # a default val in that location.
      mask = mask.flatten()
      for val in mask:
        if val:
          recon.append(default_val)
        else:
          recon.append(packed.pop(0))

      # Reshape this slice to the original (row_dim, col_dim) from the original
      # a.
      recon = np.array(recon, dtype=dtype).reshape(is_default.shape[-2:])
      a.append(recon)

    a = np.stack(a)
    a = a.reshape(is_default.shape)

    return {'a': a, 'default_val': default_val}
