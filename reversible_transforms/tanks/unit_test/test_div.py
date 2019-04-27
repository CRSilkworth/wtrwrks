import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.tanks.div as dv
import numpy as np


class TestDiv(th.TestTank):
  def test_int(self):
    self.pour_pump(dv.div, {'a': 5, 'b': 2}, {'target': 2, 'a': 5, 'b': 2})

  def test_array_int(self):
    self.pour_pump(
      dv.div,
      {'a': np.array([1, 2]), 'b': 2},
      {'target': np.array([0, 1]), 'a': np.array([1, 2]), 'b': 2},
      type_dict={'a': np.ndarray, 'b': int}
    )

  def test_int_array(self):
    self.pour_pump(
      dv.div,
      {'a': 2., 'b': np.array([1, 2])},
      {'target': np.array([2., 1.]), 'a': 2, 'b': np.array([1, 2])},
      type_dict={'a': int, 'b': np.ndarray}
    )

  def test_array(self):
    self.pour_pump(
      dv.div,
      {'a': np.array([1., 2.]), 'b': np.array([3., 4.])},
      {'target': np.array([1./3., 0.5]), 'a': np.array([1., 2.]), 'b': np.array([3., 4.])},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    self.pour_pump(
      dv.div,
      {'a': np.array([2]), 'b': np.array([3, 4])},
      {'target': np.array([0, 0]), 'a': np.array([2]), 'b': np.array([3, 4])},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    self.pour_pump(
      dv.div,
      {'b': np.array([0]), 'a': np.array([3., 4.])},
      {'target': np.array([np.inf, np.inf]), 'b': np.array([0]), 'a': np.array([3, 4])},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

if __name__ == "__main__":
    unittest.main()
