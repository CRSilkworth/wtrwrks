import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import numpy as np


def cast(a, dtype, type_dict=None, waterwork=None, name=None):
  """Find the min of a np.array along one or more axes in a reversible manner.

  Parameters
  ----------
  a : Tube, np.ndarray or None
      The array to get the min of.
  dtype : Tube, int, tuple or None
      The dtype (axes) along which to take the min.
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
  type_dict = ut.infer_types(type_dict, a=a, dtype=dtype)

  if type_dict['a'] is np.ndarray:

    class CastNPTyped(CastNP):
      slot_keys = ['a', 'dtype']
      tube_dict = {
        'target': dtype,
        'input_dtype': type(type_dict['a'].dtype),
        'diff': np.ndarray
      }

    return CastNPTyped(a=a, dtype=dtype, waterwork=waterwork, name=name)
  else:
    class CastBasicTyped(CastBasic):
      slot_keys = ['a', 'dtype']
      tube_dict = {
        'target': dtype,
        'input_dtype': type(type_dict['dtype']),
        'diff': np.ndarray
      }

    return CastBasicTyped(a=a, dtype=dtype, waterwork=waterwork, name=name)

class Cast(ta.Tank):
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

  slot_keys = ['a', 'dtype']
  tube_dict = {
    'target': np.ndarray,
    'input_dtype': type,
    'diff': np.ndarray
  }
class CastBasic(Cast):
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

  slot_keys = ['a', 'dtype']
  tube_dict = {
    'target': None,
    'input_dtype': None,
    'diff': np.ndarray
  }

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
    if dtype is not np.ndarray:
      target = dtype(a)
    else:
      target = np.ndarray(a)

    if type(a) is float and dtype in (int, bool):
      diff = a - target
    else:
      diff = 0
    # Must just return 'a' as well since so much information is lost in a
    # min
    return {'target': target, 'diff': diff, 'input_dtype': type(a)}

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

    dtype = type(target)
    if diff:
      a = diff + target
    else:
      a = input_dtype(target)
    return {'a': a, 'dtype': dtype}


class CastNP(Cast):
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

  slot_keys = ['a', 'dtype']
  tube_dict = {
    'target': None,
    'input_dtype': None,
    'diff': np.ndarray
  }

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
    target = a.astype(dtype)
    if a.dtype in (np.float64, np.float32) and dtype in (np.int32, np.int64, np.bool):
      diff = a - target
    else:
      diff = np.zeros([0])
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
    if diff.size:
      a = diff + target
    else:
      a = target.astype(input_dtype)
    return {'a': a, 'dtype': dtype}
