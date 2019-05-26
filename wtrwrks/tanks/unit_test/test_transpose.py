import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np


class TestTranspose(th.TestTank):

  def test_two_d(self):
    a = np.array([[0, 1], [2, 3], [4, 5]])
    self.pour_pump(
      td.transpose,
      {
        'a': a,
        'axes': [1, 0]
      },
      {
        'target': np.array([[0, 2, 4], [1, 3, 5]]),
        'axes': [1, 0]
      },
      type_dict={'a': np.ndarray, 'axes': tuple}
    )

  def test_three_d(self):
    a = np.arange(24).reshape((4, 3, 2, 1))
    self.pour_pump(
      td.transpose,
      {
        'a': a,
        'axes': (0, 1, 3, 2)
      },
      {
        'target': np.transpose(a, (0, 1, 3, 2)),
        'axes': (0, 1, 3, 2)
      },
      type_dict={'a': np.ndarray, 'axes': tuple}
    )
    a = np.arange(24).reshape((4, 3, 2, 1))
    self.pour_pump(
      td.transpose,
      {
        'a': a,
        'axes': (3, 0, 1, 2)
      },
      {
        'target': np.transpose(a, (3, 0, 1, 2)),
        'axes': (3, 0, 1, 2)
      },
      type_dict={'a': np.ndarray, 'axes': tuple}
    )

    self.pour_pump(
      td.transpose,
      {
        'a': a,
        'axes': (0, 3, 1, 2)
      },
      {
        'target': np.transpose(a, (0, 3, 1, 2)),
        'axes': (0, 3, 1, 2)
      },
      type_dict={'a': np.ndarray, 'axes': tuple}
    )


if __name__ == "__main__":
    unittest.main()
