import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.tanks.add as ad
import numpy as np


class TestAdd(th.TestTank):
  def test_int(self):
    self.pour_pump(ad.add, {'a': 1, 'b': 2}, {'target': 3, 'a': 1})

  def test_array_int(self):
    self.pour_pump(
      ad.add,
      {'a': np.array([1, 2]), 'b': 2},
      {'target': np.array([3, 4]), 'non_array': 2, 'a_array': True},
      type_dict={'a': np.ndarray, 'b': int}
    )

  def test_int_array(self):
    self.pour_pump(
      ad.add,
      {'a': 2, 'b': np.array([1, 2])},
      {'target': np.array([3, 4]), 'non_array': 2, 'a_array': False},
      type_dict={'a': int, 'b': np.ndarray}
    )

  def test_array(self):
    self.pour_pump(
      ad.add,
      {'a': np.array([1, 2]), 'b': np.array([3, 4])},
      {'target': np.array([4, 6]), 'smaller': np.array([3, 4]), 'a_smaller': False},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    self.pour_pump(
      ad.add,
      {'a': np.array([2]), 'b': np.array([3, 4])},
      {'target': np.array([5, 6]), 'smaller': np.array([2]), 'a_smaller': True},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

if __name__ == "__main__":
    unittest.main()
