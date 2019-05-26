import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np


class TestSplit(th.TestTank):

  def test_one_d(self):
    self.pour_pump(
      td.split,
      {
        'a': np.array([1, 2, 3]),
        'indices': np.array([1, 2]),
        'axis': 0
      },
      {
        'target': [np.array([1]), np.array([2]), np.array([3])],
        'indices': np.array([1, 2]),
        'axis': 0
      },
      type_dict={'a': np.ndarray, 'indices': np.ndarray, 'axis': int}
    )
    self.pour_pump(
      td.split,
      {
        'a': np.array([1, 2, 3]),
        'indices': np.array(3),
        'axis': 0
      },
      {
        'target': [np.array([1]), np.array([2]), np.array([3])],
        'indices': np.array(3),
        'axis': 0
      },
      type_dict={'a': np.ndarray, 'indices': np.ndarray, 'axis': int}
    )

  def test_two_d(self):
    a = np.array([[0, 1], [2, 3], [4, 5], [1, 0]])
    self.pour_pump(
      td.split,
      {
        'a': a,
        'indices': np.array([2, 4]),
        'axis': 0
      },
      {
        'target': np.split(a, np.array([2, 4])),
        'indices': np.array([2, 4]),
        'axis': 0
      },
      type_dict={'a': np.ndarray, 'indices': np.ndarray, 'axis': int}
    )
    a = np.array([[0, 1], [2, 3], [4, 5], [1, 0]])
    self.pour_pump(
      td.split,
      {
        'a': a,
        'indices': [1, 2],
        'axis': 1
      },
      {
        'target': np.split(a, [1, 2], axis=1),
        'indices': [1, 2],
        'axis': 1
      },
      type_dict={'a': np.ndarray, 'indices': np.ndarray, 'axis': int}
    )


if __name__ == "__main__":
    unittest.main()
