import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import numpy as np


class OneHot(ta.Tank):
  """The base OneHot class. All subclasses must have the same outputs where they output a numpy arrays of 'target' and 'missing_vals'. The indices must be an integer type (or array of integers) and the depth must be an int.

  Attributes
  ----------
  slot_keys : list of str
    The tank's (operation's) argument keys. They define the names of the inputs to the tank.
  tube_keys : dict(
    keys - strs. The tank's (operation's) output keys. THey define the names of the outputs of the tank
    values - types. The types of the arguments outputs.
  )
    The tank's (operation's) output keys and their corresponding types.

  """

  slot_keys = ['indices', 'depth']
  tube_keys = ['target', 'missing_vals']

  def _pour(self, indices, depth):
    indices = np.array(indices)
    # Pull out all the indices which are not in the range 0 <= index <
    # depth.
    missing_vals = indices[(indices < 0) | (indices >= depth)]

    # Reshape the indices to give them a new dimension. Compare them to
    # each index between 0 and depth - 1, find the location where they are
    # True, and turn the Trues into 1's
    indices = np.expand_dims(indices, axis=-1)
    target = (np.arange(depth) == indices).astype(np.float64)

    return {'target': target, 'missing_vals': missing_vals}

  def _pump(self, target, missing_vals):
    # Start the indices all as -1.
    indices = -1 * np.ones(target.shape[:-1], missing_vals.dtype)
    depth = int(target.shape[-1])

    # If the target is a one dimensional array, then just set to either the
    # missing val in the case that it's a zero hot or to the index of the
    # non zero value.
    if len(target.shape) == 1:
      if missing_vals.size:
        indices = missing_vals[0]
      else:
        indices = np.where(target > 0)[0][0]
    # If the target is more than on dimensional, then first find all the
    # locations where there are non-zeros. Use those locations to build an
    # array of shape target.shape[:-1] and use the last dimension of the
    # 'where' array to set the the index. Replace all the -1's with the
    # missing vals.
    else:
      unpacked_indices = np.where(target > 0)
      locs = unpacked_indices[:-1]
      vals = unpacked_indices[-1]

      indices[locs] = vals
      indices[indices == -1] = missing_vals

    return {'indices': indices, 'depth': depth}
