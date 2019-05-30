import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np


class TestCatToIndex(th.TestTank):
  def test_int(self):
    cat_to_index_map = {1: 0, 2: 1, 3: 2}
    self.pour_pump(td.cat_to_index, {'cats': 1, 'cat_to_index_map': cat_to_index_map}, {'target': 0, 'missing_vals': 0, 'cat_to_index_map': cat_to_index_map, 'input_dtype': int}, type_dict={'cats': int, 'cat_to_index_map': dict})

  def test_float(self):
    cat_to_index_map = {1.: 0, 2.: 1, 3.: 2}
    self.pour_pump(td.cat_to_index, {'cats': 1., 'cat_to_index_map': cat_to_index_map}, {'target': 0, 'missing_vals': 0.0, 'cat_to_index_map': cat_to_index_map, 'input_dtype': float}, type_dict={'cats': int, 'cat_to_index_map': dict})

  def test_str(self):
    cat_to_index_map = {'a': 0, 'b': 1, 'c': 2}
    self.pour_pump(
      td.cat_to_index,
      {'cats': 'b', 'cat_to_index_map': cat_to_index_map},
      {'target': 1, 'missing_vals': '', 'cat_to_index_map': cat_to_index_map, 'input_dtype': str},
      type_dict={'cats': str, 'cat_to_index_map': dict}
    )

    cat_to_index_map = {'a': 0, 'b': 1, 'c': 2}
    self.pour_pump(td.cat_to_index, {'cats': 'd', 'cat_to_index_map': cat_to_index_map}, {'target': -1, 'missing_vals': 'd', 'cat_to_index_map': cat_to_index_map, 'input_dtype': str}, type_dict={'cats': str, 'cat_to_index_map': dict})

  def test_scalar(self):
    cat_to_index_map = {'a': 0, 'b': 1, 'c': 2}
    self.pour_pump(
      td.cat_to_index,
      {'cats': np.array('d'), 'cat_to_index_map': cat_to_index_map},
      {'target': np.array(-1), 'missing_vals': 'd', 'cat_to_index_map': cat_to_index_map, 'input_dtype': np.object},
      type_dict={'cats': np.ndarray, 'cat_to_index_map': dict}
    )

  def test_one_d(self):
    cat_to_index_map = {0.0: 0, 1.5: 1, 3.0: 2}
    self.pour_pump(
      td.cat_to_index,
      {'cats': np.array([2., 3.]), 'cat_to_index_map': cat_to_index_map},
      {'target': np.array([-1, 2]), 'missing_vals': [2.0, 0.0], 'cat_to_index_map': cat_to_index_map, 'input_dtype': np.float64},
      type_dict={'cats': np.ndarray, 'cat_to_index_map': dict}
    )

  def test_two_d(self):
    cat_to_index_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    self.pour_pump(
      td.cat_to_index,
      {'cats': np.array([['a', 'b'], ['c', 'd'], ['e', 'f'], ['a', 'h']]), 'cat_to_index_map': cat_to_index_map},
      {
        'target': np.array([[0, 1], [2, 3], [-1, -1], [0, -1]]),
        'missing_vals': [['', ''], ['', ''], ['e', 'f'], ['', 'h']],
        'cat_to_index_map': cat_to_index_map,
        'input_dtype': np.object
      },
      type_dict={'cats': np.ndarray, 'cat_to_index_map': dict}
    )

if __name__ == "__main__":
    unittest.main()
