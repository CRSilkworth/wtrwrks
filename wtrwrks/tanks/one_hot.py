"""OneHot tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import numpy as np


class OneHot(ta.Tank):
  """The defintion of the OneHot tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'one_hot'
  slot_keys = ['indices', 'depth']
  tube_keys = ['target', 'missing_vals']

  def _pour(self, indices, depth):
    """Execute the OneHot tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    indices: np.ndarray of ints
      The array of indices to be one hotted.
    depth: int
      The maximum allowed index value and the size of the n + 1 dimension of the outputted array.

    Returns
    -------
    dict(
      target: np.ndarray
        The one hotted array.
      missing_vals: list of ints
        The indices which were not in the range of 0 <= i < depth
    )

    """
    indices = np.array(indices)
    if not indices.shape:
      target = np.zeros([depth], dtype=np.float64)
      missing_vals = -2
      if indices < 0 or indices >= depth:
        missing_vals = indices
      else:
        target[indices] = 1.0
      return {'target': target, 'missing_vals': missing_vals}
    # Pull out all the indices which are not in the range 0 <= index <
    # depth.
    mask = (indices < 0) | (indices >= depth)

    missing_vals = np.ones(indices.shape, dtype=int) * -2
    missing_vals[mask] = indices[mask]
    # Reshape the indices to give them a new dimension. Compare them to
    # each index between 0 and depth - 1, find the location where they are
    # True, and turn the Trues into 1's
    indices = np.expand_dims(indices, axis=-1)
    target = (np.arange(depth) == indices).astype(np.float64)

    return {'target': target, 'missing_vals': missing_vals}

  def _pump(self, target, missing_vals):
    """Execute the OneHot tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray
      The one hotted array.
    missing_vals: list of ints
      The indices which were not in the range of 0 <= i < depth

    Returns
    -------
    dict(
      indices: np.ndarray of ints
        The array of indices to be one hotted.
      depth: int
        The maximum allowed index value and the size of the n + 1 dimension of the outputted array.
    )

    """
    if len(target.shape) == 1:
      indices = missing_vals if not np.where(target > 0)[0] else np.where(target > 0)[0][0]

      return {'indices': indices, 'depth': target.shape[0]}
    # Start the indices all as -1.
    indices = -1 * np.ones(target.shape[:-1], missing_vals.dtype)
    depth = int(target.shape[-1])

    # If the target is a one dimensional array, then just set to either the
    # missing val in the case that it's a zero hot or to the index of the
    # non zero value.

    # If the target is more than on dimensional, then first find all the
    # locations where there are non-zeros. Use those locations to build an
    # array of shape target.shape[:-1] and use the last dimension of the
    # 'where' array to set the the index. Replace all the -1's with the
    # missing vals.
    unpacked_indices = np.where(target > 0)
    locs = unpacked_indices[:-1]
    vals = unpacked_indices[-1]

    indices[locs] = vals
    mask = indices == -1
    indices[mask] = missing_vals[mask]
    return {'indices': indices, 'depth': depth}
