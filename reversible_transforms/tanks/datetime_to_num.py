import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import numpy as np
import datetime


class DatetimeToNum(ta.Tank):
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

  slot_keys = ['a', 'zero_datetime', 'num_units', 'time_unit']
  tube_keys = ['target', 'zero_datetime', 'num_units', 'time_unit', 'diff']

  def _pour(self, a, zero_datetime, num_units, time_unit):
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
    a = np.array(a, dtype=np.datetime64)
    zero_datetime = np.array(zero_datetime, dtype=np.datetime64)
    target = (a - zero_datetime)/np.timedelta64(num_units, time_unit)

    # Save the diff since information is lost depending on the size of time unit
    undone = target * np.timedelta64(num_units, time_unit) + zero_datetime
    diff = a - undone
    if np.array(diff == np.array(0, dtype='timedelta64[us]')).all():
      diff = np.array([], dtype='timedelta64[us]')

    # Must just return 'a' as well since so much information is lost in a
    # min
    return {'target': target, 'zero_datetime': zero_datetime, 'num_units': num_units, 'time_unit': time_unit, 'diff': diff}

  def _pump(self, target, zero_datetime, num_units, time_unit, diff):
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
    a = target * np.timedelta64(num_units, time_unit) + zero_datetime
    if diff.size:
       a = a + diff

    return {'a': a, 'zero_datetime': zero_datetime, 'num_units': num_units, 'time_unit': time_unit}
