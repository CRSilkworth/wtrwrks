import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np


class TestOneHot(th.TestTank):
  def test_int(self):
    self.pour_pump(td.one_hot, {'indices': 1, 'depth': 2}, {'target': np.array([0., 1.]), 'missing_vals': -2}, type_dict={'indices': int, 'depth': int}, test_type=False)

    self.pour_pump(td.one_hot, {'indices': 3, 'depth': 2}, {'target': np.array([0, 0]), 'missing_vals': 3}, type_dict={'indices': int, 'depth': int}, test_type=False)

  def test_scalar(self):
    self.pour_pump(
      td.one_hot,
      {'indices': np.array(3), 'depth': 2},
      {'target': np.array([0., 0.]), 'missing_vals': 3},
      type_dict={'indices': np.ndarray, 'depth': int},
      test_type=False
    )

  def test_one_d(self):
    self.pour_pump(
      td.one_hot,
      {'indices': np.array([2, 3]), 'depth': 3},
      {'target': np.array([[0., 0., 1.], [0, 0, 0]]), 'missing_vals': np.array([-2, 3])},
      type_dict={'indices': np.ndarray, 'depth': int}
    )

  def test_two_d(self):
    self.pour_pump(
      td.one_hot,
      {'indices': np.array([[0, 1], [2, 3], [4, 5], [1, 0]]), 'depth': 3},
      {
        'target': np.array(
          [
            [[1., 0, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 0, 0]],
          ]
        ),
        'missing_vals': np.array([[-2, -2], [-2, 3], [4, 5], [-2, -2]])
      },
      type_dict={'indices': np.ndarray, 'depth': int}
    )

if __name__ == "__main__":
    unittest.main()
