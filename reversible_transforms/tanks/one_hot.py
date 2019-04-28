import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import numpy as np


def one_hot(indices, depth, type_dict=None, waterwork=None, name=None):
  """One hotify an index or indices, keeping track of the missing values (i.e. indices outside of range 0 <= index < depth) so that it can be undone. A new dimension will be added to the indices dimension so that 'target' with have a rank one greater than 'indices'. The new dimension is always added to the end. So indices.shape == target.shape[:-1] and target.shape[-1] == depth.

  Parameters
  ----------
  indices : int or numpy array of dtype int
      The indices to be one hotted.
  depth : int
      The length of the one hotted dimension.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).
  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  type_dict = ut.infer_types(type_dict, indices=indices, depth=depth)

  if type_dict['depth'] is not int:
    raise TypeError("depth must be of type int.")

  if type_dict['indices'] is int:
    return OneHotInt(indices=indices, depth=depth, waterwork=waterwork, name=name)
  elif type_dict['indices'] is np.ndarray:
    return OneHotNP(indices=indices, depth=depth, waterwork=waterwork, name=name)
  else:
    raise TypeError(type_dict['indices'] + "is not supported.")


class OneHot(ta.Tank):
  """The base OneHot class. All subclasses must have the same outputs where they output a numpy arrays of 'target' and 'missing_vals'. The indices must be an integer type (or array of integers) and the depth must be an int.

  Attributes
  ----------
  slot_keys : list of str
    The tank's (operation's) argument keys. They define the names of the inputs to the tank.
  tube_dict : dict(
    keys - strs. The tank's (operation's) output keys. THey define the names of the outputs of the tank
    values - types. The types of the arguments outputs.
  )
    The tank's (operation's) output keys and their corresponding types.

  """

  slot_keys = ['indices', 'depth']
  tube_dict = {
    'target': np.ndarray,
    'missing_vals': np.ndarray
  }


class OneHotInt(OneHot):
  """The OneHot class where the indices input is an int."""

  def _pour(self, indices, depth):
    # Create an array of zeros of size depth.
    target = np.zeros([depth], np.float64)

    # If the index is outside the allowed indices, save it in the
    # missing_vals. Otherwise one hot the corresponding index. And keep
    # missing_vals empty.
    if indices >= 0 and indices < depth:
      target[indices] = 1
      missing_vals = np.array([], dtype=np.int64)
    else:
      missing_vals = np.array([indices])

    return {'target': target, 'missing_vals': missing_vals}

  def _pump(self, target, missing_vals):
    # Get the depth from the dimension of the array
    depth = int(target.shape[0])

    # If the missing vals isn't empty set indices to that, otherwise find
    # the location non-zero value in target and set indices to that.
    if missing_vals.size:
      indices = missing_vals[0]
    else:
      indices = int(np.where(target > 0)[0][0])

    return {'indices': indices, 'depth': depth}


class OneHotNP(OneHot):
  """The OneHot class where the indices input is an numpy array. Handles any rank for 'indices'"""
  def _pour(self, indices, depth):

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
    indices = -1 * np.ones(target.shape[:-1], np.int64)
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
