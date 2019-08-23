import shutil
import tempfile
import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.transforms.cat_transform as n
import numpy as np
import pandas as pd


class TestCatTransform(th.TestTransform):
  def setUp(self):
      self.temp_dir = tempfile.mkdtemp()
      self.array = np.array([
        ['a', 'b', 1.0],
        ['b', 'None', 2.0],
        ['c', 'b', np.nan],
        ['a', 'c', 1.0],
      ], dtype=np.object)
      self.index_to_cat_val = ['a', 'b']

  def tearDown(self):
      shutil.rmtree(self.temp_dir)

  def test_no_norm(self):
    trans = n.CatTransform(
      name='cat',
      index_to_cat_val=np.unique(self.array[:, 0: 1].astype(np.str))
    )
    trans.calc_global_values(self.array[:, 0: 1].astype(np.str))
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
        self.array[:, 0: 1].astype(np.str),
        {
          'cat/missing_vals': [[''], [''], [''], ['']],
          'cat/one_hots': target,
          'cat/indices': indices
        }
      )
      trans = self.write_read(trans, self.temp_dir)

  def test_two_cols(self):
    trans = n.CatTransform(
      name='cat',
      index_to_cat_val=np.unique(self.array[:, 0: 2].astype(np.str))
    )
    array_iter = [self.array[0:2, 0: 2].astype(str), self.array[2: 4, 0: 2].astype(str)]

    trans.calc_global_values(data_iter=array_iter)
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
        self.array[:, 0: 2].astype(np.str),
        {
          'cat/missing_vals': np.array([['', ''], ['', ''], ['', ''], ['', '']], dtype='|S4'),
          'cat/one_hots': target,
          'cat/indices': indices
        }
      )
      trans = self.write_read(trans, self.temp_dir)

  def test_df_iter(self):
    trans = n.CatTransform(
      name='cat',
      index_to_cat_val=np.unique(self.array[:, 0: 2].astype(np.str)),
      input_dtype=np.dtype('S')
    )
    array_iter = [self.array[0:2, 0: 2].astype(str), self.array[2: 4, 0: 2].astype(str)]
    df_iter = []
    for array in array_iter:
      df = pd.DataFrame(data=array, index=[0, 1], columns=['a', 'b'])
      df_iter.append(df)

    trans.calc_global_values(data_iter=df_iter)
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
        self.array[:, 0: 2].astype(np.str),
        {
          'cat/missing_vals': np.array([['', ''], ['', ''], ['', ''], ['', '']], dtype='|S4'),
          'cat/one_hots': target,
          'cat/indices': indices
        }
      )
      trans = self.write_read(trans, self.temp_dir)

  def test_index_to_cat_val(self):
    trans = n.CatTransform(
      name='cat',
      index_to_cat_val=['a', 'b']
    )
    trans.calc_global_values(self.array[:, 0: 1].astype(np.str))
    target = np.array([[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 0.0]], [[1.0, 0.0]]]).astype(float)
    indices = np.array([[0], [1], [-1], [0]])
    for i in xrange(2):
      self.pour_pump(
        trans,
        self.array[:, 0: 1].astype(np.str),
        {
          'cat/missing_vals': [[''], [''], ['c'], ['']],
          'cat/one_hots': target,
          'cat/indices': indices
        }
      )
      trans = self.write_read(trans, self.temp_dir)

  def test_null(self):
    trans = n.CatTransform(
      name='cat',
      index_to_cat_val=np.unique(self.array[:, 2: 3].astype(np.float64))
    )
    trans.calc_global_values(self.array[:, 2: 3].astype(np.float64))
    target = np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]]]).astype(float)
    indices = np.array([[0], [1], [2], [0]])

    for i in xrange(2):
      self.pour_pump(
        trans,
        self.array[:, 2: 3].astype(np.float64),
        {
          'cat/missing_vals': np.array([[0.0], [0.0], [0.0], [0.0]], dtype=float),
          'cat/one_hots': target,
          'cat/indices': indices
        }
      )
      trans = self.write_read(trans, self.temp_dir)

  def test_mean_std(self):
    trans = n.CatTransform(
      name='cat',
      norm_mode='mean_std',
      index_to_cat_val=sorted(np.unique(self.array[:, 1: 2].astype(np.str)))
    )

    array_iter = [self.array[0:2, 1: 2].astype(str), self.array[2: 4, 1: 2].astype(str)]

    trans.calc_global_values(data_iter=array_iter)
    target = np.array([
          [[0., 1, 0]],
          [[1, 0, 0]],
          [[0, 1, 0]],
          [[0, 0, 1]],
        ]).astype(float)
    # print trans.mean
    target = (target - trans.mean)/trans.std
    indices = np.array([[1], [0], [1], [2]])
    for i in xrange(2):
      self.pour_pump(
        trans,
        self.array[:, 1: 2].astype(np.str),
        {
          'cat/missing_vals': np.array([[''], [''], [''], ['']], dtype='|S4'),
          'cat/one_hots': target,
          'cat/indices': indices
        }
      )
      trans = self.write_read(trans, self.temp_dir)

  # def test_read_write(self):
  #
  #   for i in xrange(3):
  #     if i in (0, 1):
  #       array = self.array[:, i: i + 1].astype(np.str)
  #     else:
  #       array = self.array[:, i: i + 1].astype(np.float)
  #     trans = n.CatTransform(
  #       name='cat',
  #       norm_mode='mean_std',
  #       index_to_cat_val=np.unique(array)
  #     )
  #     trans.calc_global_values(array)
  #     self.write_read_example(trans, array, self.temp_dir)

  def test_errors(self):
    with self.assertRaises(ValueError):
      trans = n.CatTransform(
        norm_mode='whatever',
        name='cat',
        index_to_cat_val=['a']
      )

    with self.assertRaises(TypeError):
      trans = n.CatTransform(
        norm_mode='min_max',
        name='cat',
      )

    trans = n.CatTransform(
      name='cat',
      norm_mode='mean_std',
      index_to_cat_val=['a']
    )
    with self.assertRaises(AssertionError):
      trans.pour(np.array([1]))


if __name__ == "__main__":
    unittest.main()
