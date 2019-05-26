import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np


class TestSub(th.TestTank):
  def test_int(self):
    self.pour_pump(td.sub, {'a': 1, 'b': 2}, {'target': -1, 'smaller_size_array': 2, 'a_is_smaller': False})

  def test_array_int(self):
    self.pour_pump(
      td.sub,
      {'a': np.array([1, 2]), 'b': 2},
      {'target': np.array([-1, 0]), 'smaller_size_array': 2, 'a_is_smaller': False},
      type_dict={'a': np.ndarray, 'b': int}
    )

  def test_int_array(self):
    self.pour_pump(
      td.sub,
      {'a': 2, 'b': np.array([1, 2])},
      {'target': np.array([1, 0]), 'smaller_size_array': 2, 'a_is_smaller': True},
      type_dict={'a': int, 'b': np.ndarray}
    )

  def test_array(self):
    self.pour_pump(
      td.sub,
      {'a': np.array([1, 2]), 'b': np.array([3, 4])},
      {'target': np.array([-2, -2]), 'smaller_size_array': np.array([3, 4]), 'a_is_smaller': False},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    self.pour_pump(
      td.sub,
      {'a': np.array([2]), 'b': np.array([3, 4])},
      {'target': np.array([-1, -2]), 'smaller_size_array': np.array([2]), 'a_is_smaller': True},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    self.pour_pump(
      td.sub,
      {'a': np.array([3, 4]), 'b': np.array([2])},
      {'target': np.array([1, 2]), 'smaller_size_array': np.array([2]), 'a_is_smaller': False},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

if __name__ == "__main__":
    unittest.main()
