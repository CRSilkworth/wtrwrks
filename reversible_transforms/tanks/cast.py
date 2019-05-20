import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import numpy as np


class Cast(ta.Tank):
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

  slot_keys = ['a', 'dtype']
  tube_keys = ['target', 'input_dtype', 'diff']

  def _pour(self, a, dtype):
    """Execute the add in the pour (forward) direction .

    Parameters
    ----------
    a : np.ndarray
      The array to take the min over.
    dtype : int, tuple
      The dtype (axes) to take the min over.

    Returns
    -------
    dict(
      'target': np.ndarray
        The result of the min operation.
      'a': np.ndarray
        The original a
      'dtype': dtype
        The dtype to cast to.
    )

    """
    a = np.array(a)
    target = a.astype(dtype)

    if a.dtype in (np.float64, np.float32) and dtype in (np.int32, np.int64, np.bool, int):
      diff = a - target
    else:
      diff = np.zeros_like(target)
    # Must just return 'a' as well since so much information is lost in a
    # min
    return {'target': target, 'diff': diff, 'input_dtype': a.dtype}

  def _pump(self, target, input_dtype, diff):
    """Execute the add in the pump (backward) direction .

    Parameters
    ----------
    target: np.ndarray
      The result of the min operation.
    a : np.ndarray
      The array to take the min over.
    dtype : type
      The dtype to cast to.

    Returns
    -------
    dict(
      'a': np.ndarray
        The original a
      'dtype': in, tuple
        The dtype (axes) to take the min over.
    )

    """
    dtype = target.dtype

    a = target
    if np.issubdtype(dtype, np.number):
      a = diff + target
    a = a.astype(input_dtype)
    return {'a': a, 'dtype': dtype}
