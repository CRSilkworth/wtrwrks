import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np


class TestBools(th.TestTank):

  def test_isnan(self):
    a = np.array([[0, np.nan], [2, 3], [4, np.nan], [1, 0]])
    self.pour_pump(
      td.isnan,
      {
        'a': a,
      },
      {
        'a': a,
        'target': np.isnan(a),
      },
      type_dict={'a': np.ndarray}
    )

    a = np.arange(24).reshape((4, 3, 2)).astype(np.float64)
    a[0, 0, 0] = np.nan
    a[1, 0, 0] = np.nan
    a[0, 1, 0] = np.nan
    a[0, 0, 1] = np.nan
    a[1, 1, 1] = np.nan

    self.pour_pump(
      td.isnan,
      {
        'a': a,
      },
      {
        'target': np.isnan(a),
        'a': a,
      },
      type_dict={'a': np.ndarray}
    )

  def test_equal(self):
    a = np.array([[0, 1], [2, 3], [4, 5], [1, 0]])
    b = np.array([[0, 1], [2, 3], [4, 5], [1, 0]])
    target = np.ones(a.shape).astype(bool)
    self.pour_pump(
      td.equals,
      {'a': a, 'b': b},
      {'a': a, 'b': b, 'target': target},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    a = np.arange(24).reshape((4, 3, 2)).astype(np.float64)
    b = np.arange(24).reshape((4, 3, 2)).astype(np.float64)
    target = np.ones(a.shape).astype(bool)
    self.pour_pump(
      td.equals,
      {'a': a, 'b': b},
      {'a': a, 'b': b, 'target': target},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    b[0, 0, 1] = b[0, 0, 1] + 1
    target[0, 0, 1] = False
    self.pour_pump(
      td.equals,
      {'a': a, 'b': b},
      {'a': a, 'b': b, 'target': target},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

  def test_greater(self):
    a = np.array([[0, 1], [2, 3], [4, 5], [1, 0]])
    b = np.array([[0, 1], [2, 3], [4, 5], [1, 0]]) - 1
    target = np.ones(a.shape).astype(bool)
    self.pour_pump(
      td.greater,
      {'a': a, 'b': b},
      {'a': a, 'b': b, 'target': target},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    a = np.arange(24).reshape((4, 3, 2)).astype(np.float64)
    b = np.arange(24).reshape((4, 3, 2)).astype(np.float64) - 1
    target = np.ones(a.shape).astype(bool)
    self.pour_pump(
      td.greater,
      {'a': a, 'b': b},
      {'a': a, 'b': b, 'target': target},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    b[0, 0, 1] = b[0, 0, 1] + 1
    target[0, 0, 1] = False
    self.pour_pump(
      td.greater,
      {'a': a, 'b': b},
      {'a': a, 'b': b, 'target': target},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

  def test_greater_equal(self):
    a = np.array([[0, 1], [2, 3], [4, 5], [1, 0]])
    b = np.array([[0, 1], [2, 3], [4, 5], [1, 0]])
    target = np.ones(a.shape).astype(bool)
    self.pour_pump(
      td.greater_equal,
      {'a': a, 'b': b},
      {'a': a, 'b': b, 'target': target},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )
    a = np.array([[0, 1], [2, 3], [4, 5], [1, 0]]) + 1
    b = np.array([[0, 1], [2, 3], [4, 5], [1, 0]])
    target = np.ones(a.shape).astype(bool)
    self.pour_pump(
      td.greater_equal,
      {'a': a, 'b': b},
      {'a': a, 'b': b, 'target': target},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    a = np.arange(24).reshape((4, 3, 2)).astype(np.float64)
    b = np.arange(24).reshape((4, 3, 2)).astype(np.float64)
    target = np.ones(a.shape).astype(bool)
    self.pour_pump(
      td.greater_equal,
      {'a': a, 'b': b},
      {'a': a, 'b': b, 'target': target},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    b[0, 0, 1] = b[0, 0, 1] + 1
    target[0, 0, 1] = False
    self.pour_pump(
      td.greater_equal,
      {'a': a, 'b': b},
      {'a': a, 'b': b, 'target': target},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

  def test_less(self):
    a = np.array([[0, 1], [2, 3], [4, 5], [1, 0]])
    b = np.array([[0, 1], [2, 3], [4, 5], [1, 0]]) + 1
    target = np.ones(a.shape).astype(bool)
    self.pour_pump(
      td.less,
      {'a': a, 'b': b},
      {'a': a, 'b': b, 'target': target},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    a = np.arange(24).reshape((4, 3, 2)).astype(np.float64)
    b = np.arange(24).reshape((4, 3, 2)).astype(np.float64) + 1
    target = np.ones(a.shape).astype(bool)
    self.pour_pump(
      td.less,
      {'a': a, 'b': b},
      {'a': a, 'b': b, 'target': target},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    b[0, 0, 1] = b[0, 0, 1] - 1
    target[0, 0, 1] = False
    self.pour_pump(
      td.less,
      {'a': a, 'b': b},
      {'a': a, 'b': b, 'target': target},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

  def test_less_equal(self):
    a = np.array([[0, 1], [2, 3], [4, 5], [1, 0]])
    b = np.array([[0, 1], [2, 3], [4, 5], [1, 0]])
    target = np.ones(a.shape).astype(bool)
    self.pour_pump(
      td.less_equal,
      {'a': a, 'b': b},
      {'a': a, 'b': b, 'target': target},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )
    a = np.array([[0, 1], [2, 3], [4, 5], [1, 0]]) - 1
    b = np.array([[0, 1], [2, 3], [4, 5], [1, 0]])
    target = np.ones(a.shape).astype(bool)
    self.pour_pump(
      td.less_equal,
      {'a': a, 'b': b},
      {'a': a, 'b': b, 'target': target},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    a = np.arange(24).reshape((4, 3, 2)).astype(np.float64)
    b = np.arange(24).reshape((4, 3, 2)).astype(np.float64)
    target = np.ones(a.shape).astype(bool)
    self.pour_pump(
      td.less_equal,
      {'a': a, 'b': b},
      {'a': a, 'b': b, 'target': target},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )

    b[0, 0, 1] = b[0, 0, 1] - 1
    target[0, 0, 1] = False
    self.pour_pump(
      td.less_equal,
      {'a': a, 'b': b},
      {'a': a, 'b': b, 'target': target},
      type_dict={'a': np.ndarray, 'b': np.ndarray}
    )
if __name__ == "__main__":
    unittest.main()
