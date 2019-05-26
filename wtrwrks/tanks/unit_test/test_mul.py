import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np


class TestDiv(th.TestTank):
  def test_scalar(self):
    self.pour_pump(
      td.mul,
      {'a': 1, 'b': 2},
      {
        'target': np.array(2),
        'smaller_size_array': 2,
        'a_is_smaller': False,
        'missing_vals': np.array([], dtype=int),
      },
      type_dict={'a': np.ndarray, 'b': int},
      test_type=False
    )
    self.pour_pump(
      td.mul,
      {'a': 2, 'b': 1},
      {
        'target': np.array(2),
        'smaller_size_array': 1,
        'a_is_smaller': False,
        'missing_vals': np.array([], dtype=int),
      },
      type_dict={'a': np.ndarray, 'b': int},
      test_type=False
    )

    self.pour_pump(
      td.mul,
      {'a': 0, 'b': 1},
      {
        'target': np.array(0),
        'smaller_size_array': 1,
        'a_is_smaller': False,
        'missing_vals': np.array([0], dtype=int),
      },
      type_dict={'a': np.ndarray, 'b': int},
      test_type=False
    )
    self.pour_pump(
      td.mul,
      {'a': 1, 'b': 0},
      {
        'target': np.array(0),
        'smaller_size_array': 0,
        'a_is_smaller': False,
        'missing_vals': np.array([1], dtype=int),
      },
      type_dict={'a': np.ndarray, 'b': int},
      test_type=False
    )

  def test_array_int(self):
    self.pour_pump(
      td.mul,
      {'a': np.array([1, 2]), 'b': 2},
      {
        'target': np.array([2, 4]),
        'smaller_size_array': 2,
        'a_is_smaller': False,
        'missing_vals': np.array([], dtype=int),
      },
      type_dict={'a': np.ndarray, 'b': int}
    )
    self.pour_pump(
      td.mul,
      {'b': np.array([1, 2]), 'a': 2},
      {
        'target': np.array([2, 4]),
        'smaller_size_array': 2,
        'a_is_smaller': True,
        'missing_vals': np.array([], dtype=int),
      },
      type_dict={'a': np.ndarray, 'b': int}
    )
    self.pour_pump(
      td.mul,
      {'b': np.array([1, 3]), 'a': 2},
      {
        'target': np.array([2, 6]),
        'smaller_size_array': 2,
        'a_is_smaller': True,
        'missing_vals': np.array([], dtype=int),
      },
      type_dict={'a': np.ndarray, 'b': int}
    )

  def test_int_array(self):
    self.pour_pump(
      td.mul,
      {'a': 2., 'b': np.array([1., 2.])},
      {
        'target': np.array([2., 4.]),
        'smaller_size_array': 2.,
        'a_is_smaller': True,
        'missing_vals': np.array([]),
      },
      type_dict={'a': int, 'b': np.ndarray}
    )

    self.pour_pump(
      td.mul,
      {'a': 2., 'b': np.array([1, 0])},
      {
        'target': np.array([2., 0.]),
        'smaller_size_array': 2.,
        'a_is_smaller': True,
        'missing_vals': np.array([0], dtype=int),
      },
      type_dict={'a': int, 'b': np.ndarray},
      test_type=False
    )

    self.pour_pump(
      td.mul,
      {'a': 0., 'b': np.array([1, 2])},
      {
        'target': np.array([0., 0.]),
        'smaller_size_array': 0.,
        'a_is_smaller': True,
        'missing_vals': np.array([1, 2]),
      },
      type_dict={'a': int, 'b': np.ndarray},
      test_type=False
    )

    self.pour_pump(
      td.mul,
      {'b': 0., 'a': np.array([[1, 2], [3, 4]])},
      {
        'target': np.array([[0., 0.], [0., 0.]]),
        'smaller_size_array': 0.,
        'a_is_smaller': False,
        'missing_vals': np.array([1, 2, 3, 4]),
      },
      type_dict={'a': int, 'b': np.ndarray},
      test_type=False
    )

    self.pour_pump(
      td.mul,
      {'b': 2., 'a': np.array([[1, 0], [-1, 2]])},
      {
        'target': np.array([[2, 0.0], [-2, 4.]]),
        'smaller_size_array': 2.,
        'a_is_smaller': False,
        'missing_vals': np.array([0.]),
      },
      type_dict={'a': int, 'b': np.ndarray},
      test_type=False
    )

    self.pour_pump(
      td.mul,
      {'b': 0., 'a': np.array([-1, 2])},
      {
        'target': np.array([0., 0.]),
        'smaller_size_array': 0.,
        'a_is_smaller': False,
        'missing_vals': np.array([-1, 2]),
      },
      type_dict={'a': int, 'b': np.ndarray},
      test_type=False
    )

  def test_array(self):
    self.pour_pump(
      td.mul,
      {'a': np.array([1., 2.]), 'b': np.array([3., 4.])},
      {
        'target': np.array([3., 8.]),
        'smaller_size_array': np.array([3., 4.]),
        'a_is_smaller': False,
        'missing_vals': np.array([]),
      },
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    self.pour_pump(
      td.mul,
      {'a': np.array([1.]), 'b': np.array([3., 4.])},
      {
        'target': np.array([3., 4.]),
        'smaller_size_array': np.array([1.]),
        'a_is_smaller': True,
        'missing_vals': np.array([]),
      },
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    self.pour_pump(
      td.mul,
      {'a': np.array([0., 2.]), 'b': np.array([3., 0.])},
      {
        'target': np.array([0., 0.]),
        'smaller_size_array': np.array([3., 0.]),
        'a_is_smaller': False,
        'missing_vals': np.array([0., 2.]),
      },
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )
    self.pour_pump(
      td.mul,
      {'a': np.array([0., 0.]), 'b': np.array([-3., 0.])},
      {
        'target': np.array([0.0, 0.]),
        'smaller_size_array': np.array([-3., 0.]),
        'a_is_smaller': False,
        'missing_vals': np.array([0., 0.]),
      },
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    self.pour_pump(
      td.mul,
      {'a': np.array([0.]), 'b': np.array([[3., 4.]])},
      {
        'target': np.array([[0.0, 0.0]]),
        'smaller_size_array': np.array([0.]),
        'a_is_smaller': True,
        'missing_vals': np.array([3., 4.]),
      },
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )
    self.pour_pump(
      td.mul,
      {'a': np.array([1.]), 'b': np.array([0., 4.])},
      {
        'target': np.array([0., 4.]),
        'smaller_size_array': np.array([1.]),
        'a_is_smaller': True,
        'missing_vals': np.array([0.]),
      },
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    self.pour_pump(
      td.mul,
      {'b': np.array([0.]), 'a': np.array([[3., 0.0]])},
      {
        'target': np.array([[0., 0.]]),
        'smaller_size_array': np.array([0.]),
        'a_is_smaller': False,
        'missing_vals': np.array([3.0, 0.0]),
      },
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )
    self.pour_pump(
      td.mul,
      {'b': np.array([1.]), 'a': np.array([0., 4.])},
      {
        'target': np.array([0., 4.0]),
        'smaller_size_array': np.array([1.]),
        'a_is_smaller': False,
        'missing_vals': np.array([0.]),
      },
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )


if __name__ == "__main__":
    unittest.main()
