import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.tanks.sub as su
import numpy as np


class TestSub(th.TestTank):
  def test_int(self):
    self.pour_pump(su.sub, {'a': 1, 'b': 2}, {'target': -1, 'smaller_size_array': 2, 'a_is_smaller': False})

  def test_array_int(self):
    self.pour_pump(
      su.sub,
      {'a': np.array([1, 2]), 'b': 2},
      {'target': np.array([-1, 0]), 'smaller_size_array': 2, 'a_is_smaller': False},
      type_dict={'a': np.ndarray, 'b': int}
    )

  def test_int_array(self):
    self.pour_pump(
      su.sub,
      {'a': 2, 'b': np.array([1, 2])},
      {'target': np.array([1, 0]), 'smaller_size_array': 2, 'a_is_smaller': True},
      type_dict={'a': int, 'b': np.ndarray}
    )

  def test_array(self):
    self.pour_pump(
      su.sub,
      {'a': np.array([1, 2]), 'b': np.array([3, 4])},
      {'target': np.array([-2, -2]), 'smaller_size_array': np.array([3, 4]), 'a_is_smaller': False},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    self.pour_pump(
      su.sub,
      {'a': np.array([2]), 'b': np.array([3, 4])},
      {'target': np.array([-1, -2]), 'smaller_size_array': np.array([2]), 'a_is_smaller': True},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    self.pour_pump(
      su.sub,
      {'a': np.array([3, 4]), 'b': np.array([2])},
      {'target': np.array([1, 2]), 'smaller_size_array': np.array([2]), 'a_is_smaller': False},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

if __name__ == "__main__":
    unittest.main()
