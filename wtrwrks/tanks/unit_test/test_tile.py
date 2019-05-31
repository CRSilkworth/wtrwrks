import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np


class TestTile(th.TestTank):

  def test_scalar(self):
    a = np.array(3)
    reps = (1, 2)
    target = np.array([[3, 3]])
    self.pour_pump(
      td.tile,
      {
        'a': a,
        'reps': reps
      },
      {
        'target': target,
        'old_shape': a.shape,
        'reps': reps
      },
      test_type=False
    )

  def test_one_d(self):
    a = np.array([1, 2, 3])
    reps = ()
    target = np.array([1, 2, 3])
    self.pour_pump(
      td.tile,
      {
        'a': a,
        'reps': reps
      },
      {
        'target': target,
        'old_shape': a.shape,
        'reps': reps
      },
      test_type=False
    )

    a = np.array([1, 2, 3])
    reps = (1)
    target = np.array([1, 2, 3])
    self.pour_pump(
      td.tile,
      {
        'a': a,
        'reps': reps
      },
      {
        'target': target,
        'old_shape': a.shape,
        'reps': reps
      },
      test_type=False
    )
    a = np.array([1, 2, 3])
    reps = (2, 1)
    target = np.array([[1, 2, 3], [1, 2, 3]])
    self.pour_pump(
      td.tile,
      {
        'a': a,
        'reps': reps
      },
      {
        'target': target,
        'old_shape': a.shape,
        'reps': reps
      },
      test_type=False
    )

    a = np.array([1, 2, 3])
    reps = (2, 2, 1)
    target = np.array([
      [[1, 2, 3], [1, 2, 3]],
      [[1, 2, 3], [1, 2, 3]]
    ])
    self.pour_pump(
      td.tile,
      {
        'a': a,
        'reps': reps
      },
      {
        'target': target,
        'old_shape': a.shape,
        'reps': reps
      },
      test_type=False
    )

  def test_two_d(self):
    a = np.array([[1, 2, 3], [4, 5, 6]])
    reps = (1, 2)
    target = np.array([[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]])

    self.pour_pump(
      td.tile,
      {
        'a': a,
        'reps': reps
      },
      {
        'target': target,
        'old_shape': a.shape,
        'reps': reps
      },
      test_type=False
    )

    a = np.array([[1, 2, 3], [4, 5, 6]])
    reps = (2, 1, 3)
    target = np.array([[[1, 2, 3, 1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6, 4, 5, 6]], [[1, 2, 3, 1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6, 4, 5, 6]]])

    self.pour_pump(
      td.tile,
      {
        'a': a,
        'reps': reps
      },
      {
        'target': target,
        'old_shape': a.shape,
        'reps': reps
      },
      test_type=False
    )
if __name__ == "__main__":
    unittest.main()
