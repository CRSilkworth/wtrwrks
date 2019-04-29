import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.tanks.sum as sm
import numpy as np


class TestSum(th.TestTank):

  def test_one_d(self):
    self.pour_pump(
      sm.sum,
      {'a': np.array([1, 3]), 'axis': ()},
      {'target': np.array(4), 'a': np.array([1, 3]), 'axis': None},
      type_dict={'a': np.ndarray, 'axis': int}
    )

  def test_two_d(self):
    self.pour_pump(
      sm.sum,
      {'a': np.array([[0, 1], [2, 3], [4, 5], [1, 0]]), 'axis': 1},
      {
        'target': np.array([1, 5, 9, 1]),
        'a': np.array([[0, 1], [2, 3], [4, 5], [1, 0]]),
        'axis': 1
      },
      type_dict={'a': np.ndarray, 'axis': int}
    )

    self.pour_pump(
      sm.sum,
      {'a': np.array([[0, 1], [2, 3], [4, 5], [1, 0]]), 'axis': 0},
      {
        'target': np.array([7, 9]),
        'a': np.array([[0, 1], [2, 3], [4, 5], [1, 0]]),
        'axis': 0
      },
      type_dict={'a': np.ndarray, 'axis': int}
    )

  def test_three_d(self):
    self.pour_pump(
      sm.sum,
      {'a': np.arange(24).reshape((2, 3, 4)), 'axis': (0, 1)},
      {
        'target': np.array([60, 66, 72, 78]),
        'a': np.arange(24).reshape((2, 3, 4)),
        'axis': (0, 1)
      },
      type_dict={'a': np.ndarray, 'axis': int}
    )

if __name__ == "__main__":
    unittest.main()
