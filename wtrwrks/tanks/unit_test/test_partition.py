import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np


class TestPartition(th.TestTank):

  def test_one_d(self):
    self.pour_pump(
      td.partition,
      {
        'a': np.array([0, 1, 2, 3, 4, 5, 6]),
        'ranges': np.array([[1, 2], [0, 2], [4, 5]]),
      },
      {
        'target': [np.array([1]), np.array([0, 1]), np.array([4])],
        'ranges': np.array([[1, 2], [0, 2], [4, 5]]),
        'missing_cols': np.array([2, 3, 5, 6]),
        'missing_array': np.array([2, 3, 5, 6]),
      },
    )
    self.pour_pump(
      td.partition,
      {
        'a': np.array([-1, 22, 37, 49]),
        'ranges': np.array([[3, 4], [2, 4]]),
      },
      {
        'target': [np.array([49]), np.array([37, 49])],
        'ranges': np.array([[3, 4], [2, 4]]),
        'missing_cols': np.array([0, 1]),
        'missing_array': np.array([-1, 22]),
      },
    )

  def test_two_d(self):
    a = np.array([[0, 1], [2, 3], [4, 5], [1, 0]])
    self.pour_pump(
      td.partition,
      {
        'a': a,
        'ranges': np.array([[2, 4], [1, 3]]),
      },
      {
        'target': [np.array([[4, 5], [1, 0]]), np.array([[2, 3], [4, 5]])],
        'ranges': np.array([[2, 4], [1, 3]]),
        'missing_cols': np.array([0]),
        'missing_array': np.array([[0, 1]]),
      },
    )

  def test_three_d(self):
    a = np.array([
      [[0, 1, 2, 3], [2, 3, 4, 5]],
      [[4, 5, 6, 7], [4, 5, 6, 7]],
      [[0, 1, 2, 3], [2, 3, 4, 5]],
      [[1, 2, 3, 4], [3, 4, 5, 6]],
      [[0, 1, 2, 3], [2, 3, 4, 5]]
    ])
    self.pour_pump(
      td.partition,
      {
        'a': a,
        'ranges': [[1, 2], [3, 5]],
      },
      {
        'target': [np.array([[[4, 5, 6, 7], [4, 5, 6, 7]]]), np.array([[[1, 2, 3, 4], [3, 4, 5, 6]], [[0, 1, 2, 3], [2, 3, 4, 5]]])],
        'ranges': [[1, 2], [3, 5]],
        'missing_cols': np.array([0, 2]),
        'missing_array': np.array([[[0, 1, 2, 3], [2, 3, 4, 5]], [[0, 1, 2, 3], [2, 3, 4, 5]]]),
      },
      )


if __name__ == "__main__":
    unittest.main()
