import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np


class TestMin(th.TestTank):

  def test_one_d(self):
    self.pour_pump(
      td.min,
      {'a': np.array([1, 3]), 'axis': ()},
      {'target': np.array(1), 'a': np.array([1, 3]), 'axis': ()},
      type_dict={'a': np.ndarray, 'axis': int},
      test_type=False
    )

  def test_two_d(self):
    self.pour_pump(
      td.min,
      {'a': np.array([[0, 1], [2, 3], [4, 5], [1, 0]]), 'axis': 1},
      {
        'target': np.array([0, 2, 4, 0]),
        'a': np.array([[0, 1], [2, 3], [4, 5], [1, 0]]),
        'axis': 1
      },
      type_dict={'a': np.ndarray, 'axis': int}
    )

    self.pour_pump(
      td.min,
      {'a': np.array([[0, 1], [2, 3], [4, 5], [1, 0]]), 'axis': 0},
      {
        'target': np.array([0, 0]),
        'a': np.array([[0, 1], [2, 3], [4, 5], [1, 0]]),
        'axis': 0
      },
      type_dict={'a': np.ndarray, 'axis': int}
    )

  def test_three_d(self):
    self.pour_pump(
      td.min,
      {'a': np.arange(24).reshape((2, 3, 4)), 'axis': (0, 1)},
      {
        'target': np.array([0, 1, 2, 3]),
        'a': np.arange(24).reshape((2, 3, 4)),
        'axis': (0, 1)
      },
      type_dict={'a': np.ndarray, 'axis': int}
    )

if __name__ == "__main__":
    unittest.main()
