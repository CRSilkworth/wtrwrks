import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import numpy as np

class Flatten(ta.Tank):
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

  slot_keys = ['a']
  tube_keys = ['target', 'shape']

  def _pour(self, a):
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
    shape = a.shape
    target = a.flatten()
    print target
    return {'target': target, 'shape': shape}

  def _pump(self, target, shape):
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

    return {'a': target.reshape(shape)}
