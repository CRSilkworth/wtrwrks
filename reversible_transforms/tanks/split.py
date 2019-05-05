import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import numpy as np


def split(a, indices, axis, type_dict=None, waterwork=None, name=None):
  """Split a np.array into subarrays along one axis in a reversible manner.

  Parameters
  ----------
  a : Tube, np.ndarray or None
      The array to get the min of.
  indices : np.ndarray
    The indices of the array to make the split at. e.g. For a = np.array([0, 1, 2, 3, 4]) and indices = np.array([2, 4]) you'd get target = [np.array([0, 1]), np.array([2, 3]), np.array([4])]
  axis : Tube, int, tuple or None
      The axis (axis) along which to split.
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
  type_dict = ut.infer_types(type_dict, a=a, indices=indices, axis=axis)

  return Split(a=a, indices=indices, axis=axis, waterwork=waterwork, name=name)


class Split(ta.Tank):
  """The min class. Handles 'a's of np.ndarray type.

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

  slot_keys = ['a', 'indices', 'axis']
  tube_dict = {
    'target': (list, None),
    'indices': (np.ndarray, np.int64),
    'axis': (int, None)
  }

  def _pour(self, a, indices, axis):
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

    # Must just return 'a' as well since so much information is lost in a
    # min
    return {'target': np.split(a, indices, axis=axis), 'indices': indices, 'axis': axis}

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
    a = np.concatenate(target, axis=axis)
    return {'a': a, 'indices': indices, 'axis': axis}
