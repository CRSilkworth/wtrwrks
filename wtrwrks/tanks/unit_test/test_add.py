import unittest
import wtrwrks.utils.test_helpers as th
# import wtrwrks.tanks.add as ad
import wtrwrks.tanks.tank_defs as td
import numpy as np


class TestAdd(th.TestTank):
  def test_int(self):
    self.pour_pump(td.add, {'a': 1, 'b': 2}, {'target': 3, 'smaller_size_array': 2, 'a_is_smaller': False})

  def test_array_int(self):
    self.pour_pump(
      td.add,
      {'a': np.array([1, 2]), 'b': 2},
      {'target': np.array([3, 4]), 'smaller_size_array': 2, 'a_is_smaller': False},
      type_dict={'a': np.ndarray, 'b': int}
    )

  def test_int_array(self):
    self.pour_pump(
      td.add,
      {'a': 2, 'b': np.array([1, 2])},
      {'target': np.array([3, 4]), 'smaller_size_array': 2, 'a_is_smaller': True},
      type_dict={'a': int, 'b': np.ndarray}
    )

  def test_array(self):
    self.pour_pump(
      td.add,
      {'a': np.array([1, 2]), 'b': np.array([3, 4])},
      {'target': np.array([4, 6]), 'smaller_size_array': np.array([3, 4]), 'a_is_smaller': False},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    self.pour_pump(
      td.add,
      {'a': np.array([2]), 'b': np.array([3, 4])},
      {'target': np.array([5, 6]), 'smaller_size_array': np.array([2]), 'a_is_smaller': True},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    self.pour_pump(
      td.add,
      {'b': np.array([2]), 'a': np.array([3, 4])},
      {'target': np.array([5, 6]), 'smaller_size_array': np.array([2]), 'a_is_smaller': False},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

if __name__ == "__main__":
    unittest.main()
