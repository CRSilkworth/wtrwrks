import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np


class TestDiv(th.TestTank):
  def test_scalar(self):
    self.pour_pump(
      td.div,
      {'a': 1, 'b': 2},
      {
        'target': np.array(0),
        'smaller_size_array': 2,
        'a_is_smaller': False,
        'missing_vals': np.array([], dtype=int),
        'remainder': np.array(1),
      },
      type_dict={'a': np.ndarray, 'b': int},
      test_type=False
    )
    self.pour_pump(
      td.div,
      {'a': 2, 'b': 1},
      {
        'target': np.array(2),
        'smaller_size_array': 1,
        'a_is_smaller': False,
        'missing_vals': np.array([], dtype=int),
        'remainder': np.array(0),
      },
      type_dict={'a': np.ndarray, 'b': int},
      test_type=False
    )

    self.pour_pump(
      td.div,
      {'a': 0, 'b': 1},
      {
        'target': np.array(0),
        'smaller_size_array': 1,
        'a_is_smaller': False,
        'missing_vals': np.array([], dtype=int),
        'remainder': np.array(0),
      },
      type_dict={'a': np.ndarray, 'b': int},
      test_type=False
    )
    with self.assertRaises(ZeroDivisionError):
      self.pour_pump(
        td.div,
        {'a': 1, 'b': 0},
        {
          'target': np.array(0),
          'smaller_size_array': 0,
          'a_is_smaller': False,
          'missing_vals': np.array([], dtype=int),
          'remainder': np.array(0),
        },
        type_dict={'a': np.ndarray, 'b': int},
        test_type=False
      )

  def test_array_int(self):
    self.pour_pump(
      td.div,
      {'a': np.array([1, 2]), 'b': 2},
      {
        'target': np.array([0, 1]),
        'smaller_size_array': 2,
        'a_is_smaller': False,
        'missing_vals': np.array([], dtype=int),
        'remainder': np.array([1, 0]),
      },
      type_dict={'a': np.ndarray, 'b': int}
    )
    self.pour_pump(
      td.div,
      {'b': np.array([1, 2]), 'a': 2},
      {
        'target': np.array([2, 1]),
        'smaller_size_array': 2,
        'a_is_smaller': True,
        'missing_vals': np.array([], dtype=int),
        'remainder': np.array([0, 0]),
      },
      type_dict={'a': np.ndarray, 'b': int}
    )
    self.pour_pump(
      td.div,
      {'b': np.array([1, 3]), 'a': 2},
      {
        'target': np.array([2, 0]),
        'smaller_size_array': 2,
        'a_is_smaller': True,
        'missing_vals': np.array([3], dtype=int),
        'remainder': np.array([0, 2]),
      },
      type_dict={'a': np.ndarray, 'b': int}
    )

  def test_int_array(self):
    self.pour_pump(
      td.div,
      {'a': 2., 'b': np.array([1., 2.])},
      {
        'target': np.array([2., 1.]),
        'smaller_size_array': 2.,
        'a_is_smaller': True,
        'missing_vals': np.array([]),
        'remainder': np.array([]),
      },
      type_dict={'a': int, 'b': np.ndarray}
    )

    self.pour_pump(
      td.div,
      {'a': 2., 'b': np.array([1, 0])},
      {
        'target': np.array([2., np.inf]),
        'smaller_size_array': 2.,
        'a_is_smaller': True,
        'missing_vals': np.array([], dtype=int),
        'remainder': np.array([]),
      },
      type_dict={'a': int, 'b': np.ndarray},
      test_type=False
    )

    self.pour_pump(
      td.div,
      {'a': 0., 'b': np.array([1, 2])},
      {
        'target': np.array([0., 0.]),
        'smaller_size_array': 0.,
        'a_is_smaller': True,
        'missing_vals': np.array([1, 2]),
        'remainder': np.array([]),
      },
      type_dict={'a': int, 'b': np.ndarray},
      test_type=False
    )

    self.pour_pump(
      td.div,
      {'b': 2., 'a': np.array([1, 2])},
      {
        'target': np.array([0.5, 1.]),
        'smaller_size_array': 2.,
        'a_is_smaller': False,
        'missing_vals': np.array([]),
        'remainder': np.array([]),
      },
      type_dict={'a': int, 'b': np.ndarray},
      test_type=False
    )

    self.pour_pump(
      td.div,
      {'b': 2., 'a': np.array([[1, 0], [-1, 2]])},
      {
        'target': np.array([[0.5, 0.0], [-0.5, 1.]]),
        'smaller_size_array': 2.,
        'a_is_smaller': False,
        'missing_vals': np.array([]),
        'remainder': np.array([]),
      },
      type_dict={'a': int, 'b': np.ndarray},
      test_type=False
    )

    self.pour_pump(
      td.div,
      {'b': 0., 'a': np.array([-1, 2])},
      {
        'target': np.array([-np.inf, np.inf]),
        'smaller_size_array': 0.,
        'a_is_smaller': False,
        'missing_vals': np.array([-1, 2]),
        'remainder': np.array([]),
      },
      type_dict={'a': int, 'b': np.ndarray},
      test_type=False
    )

  def test_array(self):
    self.pour_pump(
      td.div,
      {'a': np.array([1., 2.]), 'b': np.array([3., 4.])},
      {
        'target': np.array([1./3., 0.5]),
        'smaller_size_array': np.array([3., 4.]),
        'a_is_smaller': False,
        'missing_vals': np.array([]),
        'remainder': np.array([]),
      },
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    self.pour_pump(
      td.div,
      {'a': np.array([1.]), 'b': np.array([3., 4.])},
      {
        'target': np.array([1./3., 0.25]),
        'smaller_size_array': np.array([1.]),
        'a_is_smaller': True,
        'missing_vals': np.array([]),
        'remainder': np.array([]),
      },
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    self.pour_pump(
      td.div,
      {'a': np.array([0., 2.]), 'b': np.array([3., 0.])},
      {
        'target': np.array([0., np.inf]),
        'smaller_size_array': np.array([3., 0.]),
        'a_is_smaller': False,
        'missing_vals': np.array([2.]),
        'remainder': np.array([]),
      },
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )
    self.pour_pump(
      td.div,
      {'a': np.array([0., 0.]), 'b': np.array([-3., 0.])},
      {
        'target': np.array([0.0, np.nan]),
        'smaller_size_array': np.array([-3., 0.]),
        'a_is_smaller': False,
        'missing_vals': np.array([0.]),
        'remainder': np.array([]),
      },
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    self.pour_pump(
      td.div,
      {'a': np.array([0.]), 'b': np.array([[3., 4.]])},
      {
        'target': np.array([[0.0, 0.0]]),
        'smaller_size_array': np.array([0.]),
        'a_is_smaller': True,
        'missing_vals': np.array([3., 4.]),
        'remainder': np.array([]),
      },
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )
    self.pour_pump(
      td.div,
      {'a': np.array([1.]), 'b': np.array([0., 4.])},
      {
        'target': np.array([np.inf, 0.25]),
        'smaller_size_array': np.array([1.]),
        'a_is_smaller': True,
        'missing_vals': np.array([]),
        'remainder': np.array([]),
      },
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    self.pour_pump(
      td.div,
      {'b': np.array([0.]), 'a': np.array([[3., 0.0]])},
      {
        'target': np.array([[np.inf, np.nan]]),
        'smaller_size_array': np.array([0.]),
        'a_is_smaller': False,
        'missing_vals': np.array([3.0, 0.0]),
        'remainder': np.array([]),
      },
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )
    self.pour_pump(
      td.div,
      {'b': np.array([1.]), 'a': np.array([0., 4.])},
      {
        'target': np.array([0., 4.0]),
        'smaller_size_array': np.array([1.]),
        'a_is_smaller': False,
        'missing_vals': np.array([]),
        'remainder': np.array([]),
      },
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )


if __name__ == "__main__":
    unittest.main()
