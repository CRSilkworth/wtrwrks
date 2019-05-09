import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import numpy as np

class Partition(ta.Tank):
  """The min class. Handles 'a's of np.ndarray type.

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

  slot_keys = ['a', 'indices', 'axis']
  tube_keys = ['target', 'indices', 'axis']

  def _pour(self, a, indices):
    """Execute the add in the pour (forward) direction .

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
    a = np.array(a)
    full_cols = np.arange(a.size[0], dtype=int)

    target = []
    all_ranges = []
    for col_range in indices:
      target.append(a[col_range[0]: col_range[1]])
      all_ranges.append(np.arange(col_range[0], col_range[1]))
    all_ranges = np.stack(all_ranges, axis=0)

    missing_cols = np.setdiff1d(full_cols, all_ranges)
    missing_array = a[missing_cols]
    # Must just return 'a' as well since so much information is lost in a
    # min
    return {'target': target, 'indices': indices, 'missing_cols': missing_cols, 'missing_array': missing_array}

  def _pump(self, target, indices, axis):
    """Execute the add in the pump (backward) direction .

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
    a = np.zeros()
    return {'a': a, 'indices': indices, 'axis': axis}
