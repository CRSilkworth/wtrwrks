import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import numpy as np


class Concatenate(ta.Tank):
  """The concatenate class. Handles 'a's of np.ndarray type.

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

  slot_keys = ['a_list', 'axis']
  tube_keys = ['target', 'axis', 'indices', 'dtypes']

  def _pour(self, a_list, axis):
    """Execute the concatenate in the pour (forward) direction .

    Parameters
    ----------
    a : np.ndarray
      The array to take the min over.
    indices : np.ndarray
      The indices of the array to make the split at.
    axis : int, tuple
      The axis (axis) to take the min over.

    Returns
    -------
    dict(
      'target': np.ndarray
        The result of the min operation.
      'indices' : np.ndarray
        The indices of the array to make the split at.
      'axis': in, tuple
        The axis (axis) to take the min over.
    )

    """

    indices = []
    dtypes = []
    cursor = 0
    for a_num, a in enumerate(a_list):
      a = np.array(a)
      dtypes.append(a.dtype)
      if a_num == len(a_list) - 1:
        continue
      indices.append(cursor + a.shape[axis])
      cursor += a.shape[axis]
    # Must just return 'a' as well since so much information is lost in a
    # min
    return {'target': np.concatenate(a_list, axis=axis), 'indices': np.array(indices), 'axis': axis, 'dtypes': dtypes}

  def _pump(self, target, indices, axis, dtypes):
    """Split back into subarrays.

    Parameters
    ----------
    target: np.ndarray
      The result of the min operation.
    indices : np.ndarray
      The indices of the array to make the split at.
    axis : int, tuple
      The axis (axis) to take the min over.

    Returns
    -------
    dict(
      'a': np.ndarray
        The original a
      'indices' : np.ndarray
        The indices of the array to make the split at.
      'axis': in, tuple
        The axis (axis) to take the min over.
    )

    """
    splits = np.split(target, indices, axis=axis)
    a_list = []
    for a, dtype in zip(splits, dtypes):
      a = a.astype(dtype)
      a_list.append(a)
    return {'a_list': a_list, 'axis': axis}
