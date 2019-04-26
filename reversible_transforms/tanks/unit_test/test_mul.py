import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.tanks.mul as mu
import numpy as np


class TestMul(th.TestTank):
  def test_int(self):
    self.pour_pump(mu.mul, {'a': 3, 'b': 2}, {'target': 6, 'b': 2, 'a': 3})

  def test_array_int(self):
    self.pour_pump(
      mu.mul,
      {'a': np.array([1, 2]), 'b': 2},
      {'target': np.array([2, 4]), 'a': np.array([1, 2]), 'b': 2},
    )

  def test_array(self):
    self.pour_pump(
      mu.mul,
      {'a': np.array([1, 2]), 'b': np.array([3, 4])},
      {'target': np.array([3, 8]), 'a': np.array([1, 2]), 'b': np.array([3, 4])},
    )

    self.pour_pump(
      mu.mul,
      {'a': np.array([2]), 'b': np.array([3, 4])},
      {'target': np.array([6, 8]), 'a': np.array([2]), 'b': np.array([3, 4])},
    )

if __name__ == "__main__":
    unittest.main()
