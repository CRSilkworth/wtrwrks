import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import numpy as np


class Transpose(ta.Tank):
  """The transpose class. Handles 'a's of np.ndarray type.

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

  slot_keys = ['a', 'axes']
  tube_keys = ['target', 'axes']

  def _pour(self, a, axes):
    """Transpose the array.

    Parameters
    ----------
    a : np.ndarray
      The array to take the min over.
    axes : int, tuple
      The axes to switch.

    Returns
    -------
    dict(
      'target': np.ndarray
        The result of the min operation.
      'a': np.ndarray
        The original a
      'axes': in, tuple
        The axes to switch.
    )

    """
    # Must just return 'a' as well since so much information is lost in a
    # min
    return {'target': np.transpose(a, axes=axes), 'axes': axes}

  def _pump(self, target, axes):
    """transpose the array back to it's original shape.

    Parameters
    ----------
    target: np.ndarray
      The result of the min operation.
    a : np.ndarray
      The array to take the min over.
    axes : int, tuple
      The axes to switch.

    Returns
    -------
    dict(
      'a': np.ndarray
        The original a
      'axes': in, tuple
        The axes to switch.
    )

    """
    trans_axes = np.argsort(axes)
    a = np.transpose(target, axes=trans_axes)
    return {'a': a, 'axes': axes}
