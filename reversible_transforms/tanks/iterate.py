import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import numpy as np


class IterList(ta.Tank):
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
  tube_keys = None

  def _pour(self, a):
    """Execute the add in the pour (forward) direction .

    Parameters
    ----------
    a : np.ndarray
      The array to take the min over.
    indices : np.ndarray
      The indices of the array to make the iterate at.
    axis : int, tuple
      The axis (axis) to take the min over.

    Returns
    -------
    dict(
      'target': np.ndarray
        The result of the min operation.
      'indices' : np.ndarray
        The indices of the array to make the iterate at.
      'axis': in, tuple
        The axis (axis) to take the min over.
    )

    """

    # Must just return 'a' as well since so much information is lost in a
    # min
    r_dict = {}
    for key_num, tube_key in enumerate(self.tube_keys):
      r_dict[tube_key] = a[key_num]
    return r_dict

  def _pump(self, **kwargs):
    """Execute the add in the pump (backward) direction .

    Parameters
    ----------
    target: np.ndarray
      The result of the min operation.
    indices : np.ndarray
      The indices of the array to make the iterate at.
    axis : int, tuple
      The axis (axis) to take the min over.

    Returns
    -------
    dict(
      'a': np.ndarray
        The original a
      'indices' : np.ndarray
        The indices of the array to make the iterate at.
      'axis': in, tuple
        The axis (axis) to take the min over.
    )

    """
    a = []
    for tube_key in self.tube_keys:
      a.append(kwargs[tube_key])
    return {'a': a}


class IterDict(ta.Tank):
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
  tube_keys = None

  def _pour(self, a):
    """Execute the add in the pour (forward) direction .

    Parameters
    ----------
    a : np.ndarray
      The array to take the min over.
    indices : np.ndarray
      The indices of the array to make the iterate at.
    axis : int, tuple
      The axis (axis) to take the min over.

    Returns
    -------
    dict(
      'target': np.ndarray
        The result of the min operation.
      'indices' : np.ndarray
        The indices of the array to make the iterate at.
      'axis': in, tuple
        The axis (axis) to take the min over.
    )

    """

    # Must just return 'a' as well since so much information is lost in a
    # min
    r_dict = {}
    for tube_key in self.tube_keys:
      r_dict[tube_key] = a[tube_key]
    return r_dict

  def _pump(self, **kwargs):
    """Execute the add in the pump (backward) direction .

    Parameters
    ----------
    target: np.ndarray
      The result of the min operation.
    indices : np.ndarray
      The indices of the array to make the iterate at.
    axis : int, tuple
      The axis (axis) to take the min over.

    Returns
    -------
    dict(
      'a': np.ndarray
        The original a
      'indices' : np.ndarray
        The indices of the array to make the iterate at.
      'axis': in, tuple
        The axis (axis) to take the min over.
    )

    """
    return {'a': kwargs}
