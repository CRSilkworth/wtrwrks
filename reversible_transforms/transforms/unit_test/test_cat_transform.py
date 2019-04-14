import shutil
import tempfile
import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.transforms.cat_transform as n
import os
import pandas as pd
import numpy as np

class TestCatTransform(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.array = np.array([
          ['a', 'b', 1.0],
          ['b', None, 2.0],
          ['c', 'b', np.nan],
          ['a', 'c', 1.0],
        ])
        self.valid_cats = ['a', 'b']

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_no_norm(self):
        trans = n.CatTransform(
          col_index=0,
          name='cat'
        )
        trans.calc_global_values(self.array)
        ci = trans.col_index

        for row in self.array:
          arrays_dict = trans.forward_transform(row, verbose=False)
          val = trans.backward_transform(arrays_dict, verbose=False)

          a = np.zeros([len(trans)])
          a[trans.nan_safe_cat_val_to_index(row[trans.col_index])] = 1.

          index = -1 if row[ci] not in trans.cat_val_to_index else trans.cat_val_to_index[row[ci]]

          temp_d = {
            'is_valid': np.ones((1,), dtype=np.bool),
            'val': a,
            'cat_val': row[ci: ci + 1],
            'index':index
          }
          for key in temp_d:
            th.assert_arrays_equal(self, temp_d[key], arrays_dict[key])
          th.assert_arrays_equal(self, row[ci: ci + 1], val)

    def test_valid_cats(self):
        trans = n.CatTransform(
          col_index=0,
          name='cat',
          valid_cats=['a', 'b']
        )
        trans.calc_global_values(self.array)
        ci = trans.col_index

        for row_index, row in enumerate(self.array):
          arrays_dict = trans.forward_transform(row, verbose=False)
          val = trans.backward_transform(arrays_dict, verbose=False)

          a = np.zeros([len(trans)])
          if row_index != 2:
            a[trans.nan_safe_cat_val_to_index(row[trans.col_index])] = 1.
          index = -1 if row[ci] not in trans.cat_val_to_index else trans.cat_val_to_index[row[ci]]

          temp_d = {
            'is_valid': np.ones((1,), dtype=np.bool) if row_index != 2 else np.zeros((1,), dtype=np.bool),
            'val': a,
            'cat_val': row[ci: ci + 1],
            'index': index
          }
          for key in temp_d:
            th.assert_arrays_equal(self, temp_d[key], arrays_dict[key])
          th.assert_arrays_equal(self, row[ci: ci + 1], val)

    def test_null(self):
        trans = n.CatTransform(
          col_index=2,
          name='cat'
        )
        trans.calc_global_values(self.array)
        ci = trans.col_index

        for row_index, row in enumerate(self.array):
          arrays_dict = trans.forward_transform(row, verbose=False)
          val = trans.backward_transform(arrays_dict, verbose=False)

          a = np.zeros([len(trans)])
          a[trans.nan_safe_cat_val_to_index(row[trans.col_index])] = 1.

          index = 0 if row[ci] not in trans.cat_val_to_index else trans.cat_val_to_index[row[ci]]

          temp_d = {
            'is_valid': np.ones((1,), dtype=np.bool),
            'val': a,
            'cat_val': row[ci: ci + 1],
            'index': index
          }
          for key in temp_d:
            th.assert_arrays_equal(self, temp_d[key], arrays_dict[key])
          th.assert_arrays_equal(self, row[ci: ci + 1], val)

    def test_ignore_null(self):
        trans = n.CatTransform(
          col_index=2,
          name='cat',
          ignore_null=True
        )
        trans.calc_global_values(self.array)
        ci = trans.col_index

        for row_index, row in enumerate(self.array):
          arrays_dict = trans.forward_transform(row, verbose=False)
          val = trans.backward_transform(arrays_dict, verbose=False)

          a = np.zeros([len(trans)])
          if row_index != 2:
            a[trans.nan_safe_cat_val_to_index(row[trans.col_index])] = 1.
          index = -1 if row[ci] not in trans.cat_val_to_index else trans.cat_val_to_index[row[ci]]

          temp_d = {
            'is_valid': np.ones((1,), dtype=np.bool) if row_index != 2 else np.zeros((1,), dtype=np.bool),
            'val': a,
            'cat_val': row[ci: ci + 1],
            'index': index
          }
          for key in temp_d:
            th.assert_arrays_equal(self, temp_d[key], arrays_dict[key])
          th.assert_arrays_equal(self, row[ci: ci + 1], val)

    def test_mean_std(self):
        trans = n.CatTransform(
          col_index=1,
          name='cat',
          norm_mode='mean_std'
        )
        trans.calc_global_values(self.array)
        ci = trans.col_index

        mean = trans.mean
        std = trans.std
        for row_index, row in enumerate(self.array):
          arrays_dict = trans.forward_transform(row, verbose=False)
          val = trans.backward_transform(arrays_dict, verbose=False)

          a = np.zeros([len(trans)])
          a[trans.nan_safe_cat_val_to_index(row[trans.col_index])] = 1.
          a = (a - trans.mean)/trans.std

          index = -1 if row[ci] not in trans.cat_val_to_index else trans.cat_val_to_index[row[ci]]

          temp_d = {
            'is_valid': np.ones((1,), dtype=np.bool),
            'val': a,
            'cat_val': row[ci: ci + 1],
            'index': index
          }
          for key in temp_d:
            th.assert_arrays_equal(self, temp_d[key], arrays_dict[key])

          th.assert_arrays_equal(self, row[ci: ci + 1], val)

    def test_errors(self):
      with self.assertRaises(ValueError):
        trans = n.CatTransform(
          name='cat'
        )

      with self.assertRaises(ValueError):
        trans = n.CatTransform(
          col_index=0,
          norm_mode='whatever',
          name='cat'
        )

      with self.assertRaises(ValueError):
        trans = n.CatTransform(
          col_index=0,
          norm_mode='min_max',
          name='cat'
        )

      trans = n.CatTransform(
        col_index=0,
        name='cat',
        norm_mode='mean_std',
      )
      with self.assertRaises(AssertionError):
        trans.forward_transform(np.array([1]))
        with self.assertRaises(AssertionError):
          trans.forward_transform({})

      with self.assertRaises(ValueError):
        trans.calc_global_values(np.array([[np.nan]]))

if __name__ == "__main__":
    unittest.main()
