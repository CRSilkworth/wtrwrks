import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.tanks.one_hot as oh
import numpy as np


class TestMul(th.TestTank):

  def test_one_d(self):
    self.pour_pump(
      oh.one_hot,
      {'indices': np.array([2, 3]), 'depth': 3},
      {'target': np.array([[0, 0, 1], [0, 0, 0]]), 'missing_vals': np.array([3])},
      type_dict={'indices': np.ndarray, 'depth': int}
    )

  def test_two_d(self):
    self.pour_pump(
      oh.one_hot,
      {'indices': np.array([
        [0, 1], [2, 3], [4, 5], [1, 0]]), 'depth': 3},
      {
        'target': np.array(
          [
            [[1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 0, 0]],
          ]
        ),
        'missing_vals': np.array([3, 4, 5])
      },
      type_dict={'indices': np.ndarray, 'depth': int}
    )

if __name__ == "__main__":
    unittest.main()
