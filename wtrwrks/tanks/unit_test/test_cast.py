import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np


class TestCasts(th.TestTank):

  def test_np(self):
    a = np.array([[0, 1], [2, 3], [4, 5], [1, 0]])
    target = a.astype(float)
    self.pour_pump(
      td.cast,
      {'a': a, 'dtype': np.float64},
      {'target': target, 'diff': np.zeros(a.shape, dtype=float), 'input_dtype': a.dtype},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )
    a = np.array([[0, 1.5]])
    target = a.astype(int)
    self.pour_pump(
      td.cast,
      {'a': a, 'dtype': np.int64},
      {'target': target, 'diff': np.array([[0, 0.5]]), 'input_dtype': a.dtype},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )
    a = np.array([[0, 1.5]])
    target = a.astype(bool)
    self.pour_pump(
      td.cast,
      {'a': a, 'dtype': np.bool},
      {'target': target, 'diff': np.array([[0, 0.5]]), 'input_dtype': a.dtype},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

  def test_basic(self):
    a = 1
    target = 1.0
    self.pour_pump(
      td.cast,
      {'a': a, 'dtype': float},
      {'target': target, 'diff': np.zeros((), dtype=float), 'input_dtype': type(a)},
      type_dict={'a': int, 'dtype': type(int)},
      test_type=False
    )
    a = 1.5
    target = 1
    self.pour_pump(
      td.cast,
      {'a': a, 'dtype': int},
      {'target': target, 'diff': np.array(0.5), 'input_dtype': type(a)},
      type_dict={'a': int, 'dtype': type(int)},
      test_type=False
    )
    a = 0.5
    target = True
    self.pour_pump(
      td.cast,
      {'a': a, 'dtype': bool},
      {'target': target, 'diff': -0.5, 'input_dtype': type(a)},
      test_type=False
    )
    a = 0
    target = False
    self.pour_pump(
      td.cast,
      {'a': a, 'dtype': bool},
      {'target': target, 'diff': 0, 'input_dtype': type(a)},
      type_dict={'a': int, 'dtype': type(int)},
      test_type=False
    )
if __name__ == "__main__":
    unittest.main()
