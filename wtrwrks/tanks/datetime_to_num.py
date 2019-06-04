"""DatetimeToNum tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import numpy as np
import datetime


class DatetimeToNum(ta.Tank):
  """The defintion of the DatetimeToNum tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'datetime_to_num'
  slot_keys = ['a', 'zero_datetime', 'num_units', 'time_unit']
  tube_keys = ['target', 'zero_datetime', 'num_units', 'time_unit', 'diff']
  pass_through_keys = ['num_units', 'time_unit', 'zero_datetime']

  def _pour(self, a, zero_datetime, num_units, time_unit):
    """Execute the DatetimeToNum tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: np.ndarray of datetime64
      The array of datetimes to be converted to numbers.
    zero_datetime: datetime64
      The datetime that will be considered zero when converted to a number. All other datetimes will be relative to the this.
    num_units: int
      This along with 'time_unit' define the time resolution to convert to numbers. e.g. if 'time_unit' is 'D' (day) and 'num_units' is 2 then every two days corresponds to 1 increment in time.
    time_unit: str - from 'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as'
      This along with 'num_units' define the time resolution to convert to numbers. e.g. if 'time_unit' is 'D' (day) and 'num_units' is 2 then every two days corresponds to 1 increment in time.

    Returns
    -------
    dict(
      target: np.ndarray of
        The array of datetimes that were converted to numbers.
      zero_datetime: datetime64
        The datetime that will be considered zero when converted to a number. All other datetimes will be relative to the this.
      num_units: int
        This along with 'time_unit' define the time resolution to convert to numbers. e.g. if 'time_unit' is 'D' (day) and 'num_units' is 2 then every two days corresponds to 1 increment in time.
      time_unit: str - from 'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as'
        This along with 'num_units' define the time resolution to convert to numbers. e.g. if 'time_unit' is 'D' (day) and 'num_units' is 2 then every two days corresponds to 1 increment in time.
      diff: np.ndarray of timedelta64
        The difference between the original array 'a' and the array which lost information from taking everything according to a finite time resolution. e.g. For num_units=1 and time_unit='M' datetime(2000, 3, 4) gets mapped to datetime(2000, 3) so the diff would be a timedelta64 of 4 days.
    )

    """
    a = np.array(a, dtype=np.datetime64)
    zero_datetime = np.array(zero_datetime, dtype=np.datetime64)
    target = (a - zero_datetime)/np.timedelta64(num_units, time_unit)

    # Save the diff since information is lost depending on the size of time unit
    undone = target * np.timedelta64(num_units, time_unit) + zero_datetime
    diff = a - undone

    # If there is no diff then don't bother saving any information.
    # if np.array(diff == np.array(0, dtype='timedelta64[us]')).all():
    #   diff = np.array([], dtype='timedelta64[us]')

    return {'target': target, 'zero_datetime': zero_datetime, 'num_units': num_units, 'time_unit': time_unit, 'diff': diff}

  def _pump(self, target, zero_datetime, num_units, time_unit, diff):
    """Execute the DatetimeToNum tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray of
      The array of datetimes that were converted to numbers.
    zero_datetime: datetime64
      The datetime that will be considered zero when converted to a number. All other datetimes will be relative to the this.
    num_units: int
      This along with 'time_unit' define the time resolution to convert to numbers. e.g. if 'time_unit' is 'D' (day) and 'num_units' is 2 then every two days corresponds to 1 increment in time.
    time_unit: str - from 'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as'
      This along with 'num_units' define the time resolution to convert to numbers. e.g. if 'time_unit' is 'D' (day) and 'num_units' is 2 then every two days corresponds to 1 increment in time.
    diff: np.ndarray of timedelta64
      The difference between the original array 'a' and the array which lost information from taking everything according to a finite time resolution. e.g. For num_units=1 and time_unit='M' datetime(2000, 3, 4) gets mapped to datetime(2000, 3) so the diff would be a timedelta64 of 4 days.

    Returns
    -------
    dict(
      a: np.ndarray of datetime64
        The array of datetimes to be converted to numbers.
      zero_datetime: datetime64
        The datetime that will be considered zero when converted to a number. All other datetimes will be relative to the this.
      num_units: int
        This along with 'time_unit' define the time resolution to convert to numbers. e.g. if 'time_unit' is 'D' (day) and 'num_units' is 2 then every two days corresponds to 1 increment in time.
      time_unit: str - from 'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as'
        This along with 'num_units' define the time resolution to convert to numbers. e.g. if 'time_unit' is 'D' (day) and 'num_units' is 2 then every two days corresponds to 1 increment in time.
    )

    """

    # Get back the original multiplying by the time resolution and adding the
    # zero_date plus any diff values.
    a = target * np.timedelta64(num_units, time_unit) + zero_datetime
    if diff.size:
       a = a + diff

    return {'a': a, 'zero_datetime': zero_datetime, 'num_units': num_units, 'time_unit': time_unit}
