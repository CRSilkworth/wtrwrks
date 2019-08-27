import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np


class TestReplace(th.TestTank):

  def test_one_d(self):
    self.pour_pump(
      td.replace,
      {
        'a': np.array([1, 3]),
        'mask': np.array([True, False]),
        'replace_with': np.array([0])
      },
      {
        'target': np.array([0, 3]),
        'mask': np.array([True, False]),
        'replaced_vals': np.array([1, 0]),
        'replace_with': np.array([0])
      },
      type_dict={'a': np.ndarray, 'mask': np.ndarray, 'replace_with': np.ndarray},
      test_type=False
    )

  def test_two_d(self):
    self.pour_pump(
      td.replace,
      {
        'a': np.array([[0, 1], [2, 3], [4, 5], [1, 0]]),
        'mask': np.array([[0, 1], [1, 1], [0, 1], [1, 0]]).astype(bool),
        'replace_with': np.array([6, 7, 6, 7, 6])
      },
      {
        'target': np.array([[0, 6], [7, 6], [4, 7], [6, 0]]),
        'mask': np.array([[0, 1], [1, 1], [0, 1], [1, 0]]).astype(bool),
        'replaced_vals': np.array([[0, 1], [2, 3], [0, 5], [1, 0]]),
        'replace_with': np.array([6, 7, 6, 7, 6])
      },
      type_dict={'a': np.ndarray, 'mask': np.ndarray, 'replace_with': np.ndarray}
    )

    self.pour_pump(
      td.replace,
      {
        'a': np.array([[0, 1], [2, 3], [4, 5]]),
        'mask': np.array([True, False, False]),
        'replace_with': np.array([-1])
      },
      {
        'target': np.array([[-1, -1], [2, 3], [4, 5]]),
        'mask': np.array([True, False, False]),
        'replaced_vals': np.array([[0, 1], [0, 0], [0, 0]]),
        'replace_with': np.array([-1])
      },
      type_dict={'a': np.ndarray, 'mask': np.ndarray, 'replace_with': np.ndarray},
      test_type=False
    )

  def test_three_d(self):
    target = np.concatenate([np.arange(18).reshape((3, 3, 2)), np.zeros((1, 3, 2))], axis=0)
    self.pour_pump(
      td.replace,
      {
        'a': np.arange(24, dtype=float).reshape((4, 3, 2)),
        'mask': np.array([False, False, False, True]),
        'replace_with': np.array([0])
      },
      {
        'target': target,
        'mask': np.array([False, False, False, True]),
        'replaced_vals': np.array([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[18.0, 19.0], [20.0, 21.0], [22.0, 23.0]]], dtype=float),
        'replace_with': np.array([0])
      },
      type_dict={'a': np.ndarray, 'mask': np.ndarray, 'replace_with': np.ndarray},
      test_type=False
    )

if __name__ == "__main__":
    unittest.main()
