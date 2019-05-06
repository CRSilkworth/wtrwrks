import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.tanks.tank_defs as td
import numpy as np
from datetime import datetime, timedelta

class TestDatetimeToNum(th.TestTank):

  def test_scalar(self):
    self.pour_pump(
      td.datetime_to_num,
      {
        'a': np.array(datetime(1970, 1, 1), dtype=np.datetime64),
        'zero_datetime': np.array(datetime(1970, 1, 1), dtype=np.datetime64),
        'time_unit': 'D',
        'num_units': 1
      },
      {
        'target': np.array(0),
        'zero_datetime': np.array(datetime(1970, 1, 1), dtype=np.datetime64),
        'time_unit': 'D',
        'num_units': 1,
        'diff': np.array([], dtype='timedelta64[us]')
      },
      test_type=False
    )

  def test_one_d(self):
    a = np.array(['2019-01-01', '2019-02-01', '2019-03-01'], dtype=np.datetime64)
    diff = np.array([-1546041600000000, -1548806400000000, -1551225600000000], dtype=np.timedelta64)
    zero_datetime = np.array(datetime(1970, 1, 1), dtype=np.datetime64)

    self.pour_pump(
      td.datetime_to_num,
      {'a': a, 'num_units': 2, 'time_unit': 'W', 'zero_datetime': zero_datetime},
      {
        'target': np.array([2556.71428571, 2561.14285714, 2565.14285714]),
        'zero_datetime': np.array(datetime(1970, 1, 1), dtype=np.datetime64),
        'time_unit': 'W',
        'num_units': 2,
        'diff': diff,
      },
      # type_dict={'a': np.ndarray, 'axis': int}
      test_type=False
    )

    zero_datetime = np.array(datetime(2000, 1, 1), dtype=np.datetime64)
    diff = np.array([-59361984000000000, -59627145600000000, -59866646400000000], dtype=np.timedelta64)
    self.pour_pump(
      td.datetime_to_num,
      {'a': a, 'num_units': 100, 'time_unit': 's', 'zero_datetime': zero_datetime},
      {
        'target': np.array([5.996160e+08, 6.022944e+08, 6.047136e+08]),
        'zero_datetime': zero_datetime,
        'time_unit': 's',
        'num_units': 100,
        'diff': diff,
      },
      # type_dict={'a': np.ndarray, 'axis': int}
      test_type=False
    )

  #   self.pour_pump(
  #     td.datetime_to_num,
  #     {'a': np.array([[0, 1], [2, 3], [4, 5], [1, 0]]), 'axis': 0},
  #     {
  #       'target': np.array([1.75, 2.25]),
  #       'a': np.array([[0, 1], [2, 3], [4, 5], [1, 0]]),
  #       'axis': 0
  #     },
  #     type_dict={'a': np.ndarray, 'axis': int}
  #   )
  #
  # def test_three_d(self):
  #   self.pour_pump(
  #     td.datetime_to_num,
  #     {'a': np.arange(24).reshape((2, 3, 4)), 'axis': (0, 1)},
  #     {
  #       'target': np.array([10., 11, 12, 13]),
  #       'a': np.arange(24).reshape((2, 3, 4)),
  #       'axis': (0, 1)
  #     },
  #     type_dict={'a': np.ndarray, 'axis': int}
  #   )

if __name__ == "__main__":
    unittest.main()
