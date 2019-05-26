import shutil
import tempfile
import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.transforms.cat_transform as n
import numpy as np


class TestCatTransform(th.TestTransform):
  def setUp(self):
      self.temp_dir = tempfile.mkdtemp()
      self.array = np.array([
        ['a', 'b', 1.0],
        ['b', 'None', 2.0],
        ['c', 'b', np.nan],
        ['a', 'c', 1.0],
      ], dtype=np.object)
      self.valid_cats = ['a', 'b']

  def tearDown(self):
      shutil.rmtree(self.temp_dir)

  def test_no_norm(self):
    trans = n.CatTransform(
      name='cat'
    )
    trans.calc_global_values(self.array[:, 0: 1])
    target = np.array([
      [[1., 0., 0.]],
      [[0., 1., 0.]],
      [[0., 0., 1.]],
      [[1., 0., 0.]]
    ])
    indices = np.array([[0], [1], [2], [0]])
    for i in xrange(2):
      self.pour_pump(
        trans,
        self.array[:, 0: 1],
        {
          'cat/missing_vals': [],
          'cat/one_hots': target,
          'cat/indices': indices
        }
      )
      trans = self.write_read(trans, self.temp_dir)

  def test_two_cols(self):
    trans = n.CatTransform(
      name='cat'
    )
    trans.calc_global_values(self.array[:, 0: 2])
    target = np.array([
      [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
      [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
      [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
      [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    ])
    indices = np.array([[1, 2], [2, 0], [3, 2], [1, 3]])
    for i in xrange(2):
      self.pour_pump(
        trans,
        self.array[:, 0: 2],
        {
          'cat/missing_vals': [],
          'cat/one_hots': target,
          'cat/indices': indices
        }
      )
      trans = self.write_read(trans, self.temp_dir)

  def test_valid_cats(self):
    trans = n.CatTransform(
      name='cat',
      valid_cats=['a', 'b']
    )
    trans.calc_global_values(self.array[:, 0: 1])
    target = np.array([[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 0.0]], [[1.0, 0.0]]]).astype(float)
    indices = np.array([[0], [1], [-1], [0]])
    for i in xrange(2):
      self.pour_pump(
        trans,
        self.array[:, 0: 1],
        {
          'cat/missing_vals': ['c'],
          'cat/one_hots': target,
          'cat/indices': indices
        }
      )
      trans = self.write_read(trans, self.temp_dir)

  def test_null(self):
    trans = n.CatTransform(
      name='cat',
    )
    trans.calc_global_values(self.array[:, 2: 3])
    target = np.array([
          [[0, 1, 0]],
          [[0, 0, 1]],
          [[1, 0, 0]],
          [[0, 1, 0]]
        ]).astype(float)
    indices = np.array([[1], [2], [0], [1]])
    for i in xrange(2):
      self.pour_pump(
        trans,
        self.array[:, 2: 3],
        {
          'cat/missing_vals': np.array([], dtype=int),
          'cat/one_hots': target,
          'cat/indices': indices
        }
      )
      trans = self.write_read(trans, self.temp_dir)

  def test_ignore_null(self):
    trans = n.CatTransform(
      name='cat',
      ignore_null=True
    )
    trans.calc_global_values(self.array[:, 2: 3])
    indices = np.array([[0], [1], [-1], [0]])
    target = np.array([
      [[1, 0]],
      [[0, 1]],
      [[0, 0]],
      [[1, 0]]
    ]).astype(float)

    for i in xrange(2):
      self.pour_pump(
        trans,
        self.array[:, 2: 3],
        {
          'cat/missing_vals': [np.nan],
          'cat/one_hots': target,
          'cat/indices': indices
        }
      )
      trans = self.write_read(trans, self.temp_dir)

  def test_mean_std(self):
    trans = n.CatTransform(
      name='cat',
      norm_mode='mean_std'
    )
    trans.calc_global_values(self.array[:, 1: 2])
    target = np.array([
          [[0., 1, 0]],
          [[1, 0, 0]],
          [[0, 1, 0]],
          [[0, 0, 1]],
        ]).astype(float)
    target = (target - trans.mean)/trans.std
    indices = np.array([[1], [0], [1], [2]])
    for i in xrange(2):
      self.pour_pump(
        trans,
        self.array[:, 1: 2],
        {
          'cat/missing_vals': np.array([], dtype=int),
          'cat/one_hots': target,
          'cat/indices': indices
        }
      )
      trans = self.write_read(trans, self.temp_dir)

  def test_read_write(self):

    for i in xrange(3):
      if i in (0, 1):
        array = self.array[:, i: i + 1].astype(np.str)
      else:
        array = self.array[:, i: i + 1].astype(np.float)
      trans = n.CatTransform(
        name='cat',
        norm_mode='mean_std'
      )

      trans.calc_global_values(array)
      self.write_read_example(trans, array, self.temp_dir)

  def test_errors(self):
    with self.assertRaises(ValueError):
      trans = n.CatTransform(
        norm_mode='whatever',
        name='cat'
      )

    with self.assertRaises(ValueError):
      trans = n.CatTransform(
        norm_mode='min_max',
        name='cat'
      )

    trans = n.CatTransform(
      name='cat',
      norm_mode='mean_std',
    )
    with self.assertRaises(AssertionError):
      trans.pour(np.array([1]))


if __name__ == "__main__":
    unittest.main()
