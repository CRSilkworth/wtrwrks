import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np


class TestConcatenate(th.TestTank):

  def test_one_d(self):
    self.pour_pump(
      td.concatenate,
      {
        'a_list': [np.array([1]), np.array([2]), np.array([3])],
        'axis': 0
      },
      {
        'target': np.array([1, 2, 3]),
        'indices': np.array([1, 2]),
        'axis': 0,
        'dtypes': [np.int64, np.int64, np.int64]
      },
      type_dict={'a_list': list, 'axis': int}
    )
    self.pour_pump(
      td.concatenate,
      {
        'a_list': [np.array([1.]), np.array([2]), np.array([3])],
        'axis': 0
      },
      {
        'target': np.array([1., 2., 3.]),
        'indices': np.array([1, 2]),
        'axis': 0,
        'dtypes': [np.float64, np.int64, np.int64]
      },
      type_dict={'a_list': list, 'axis': int}
    )

  def test_two_d(self):
    a = np.array([[0, 1], [2, 3], [4, 5], [1, 0], [6, 7]])
    self.pour_pump(
      td.concatenate,

      {
        'a_list': np.split(a, np.array([2, 3, 5])),
        'axis': 0
      },
      {
        'target': a,
        'indices': np.array([2, 3, 5]),
        'axis': 0,
        'dtypes': [np.int64, np.int64, np.int64, np.int64]
      },
      type_dict={'a_list': list, 'axis': int}
    )
    a = np.array([[0, 1], [2, 3], [4, 5], [1, 0]])
    self.pour_pump(
      td.concatenate,
      {
        'a_list': np.split(a, [1, 2], axis=1),
        'axis': 1
      },
      {
        'target': a,
        'indices': [1, 2],
        'axis': 1,
        'dtypes': [np.int64, np.int64, np.int64]
      },
      type_dict={'a_list': list, 'axis': int}
    )


if __name__ == "__main__":
    unittest.main()
