import shutil
import tempfile
import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.transforms.num_transform as n
import os
import pandas as pd
import numpy as np

class TestNumTransform(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.array = np.array([
          [1, 2, np.nan],
          [4, np.nan, np.nan],
          [7, 8, np.nan],
          [10, 11, np.nan]
        ])

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_no_norm(self):
        trans = n.NumTransform(
          col_index=0,
          name='num'
        )
        trans.calc_global_values(self.array)
        ci = trans.col_index

        for row in self.array:
          arrays_dict = trans.forward_transform(row, verbose=False)
          val = trans.backward_transform(arrays_dict, verbose=False)

          self.assertEqual(
            {'is_null': np.zeros((1,), dtype=np.bool), 'val': row[ci: ci + 1]},
            arrays_dict
          )
          th.assert_arrays_equal(self, row[ci: ci + 1], val)

    def test_null(self):
        trans = n.NumTransform(
          col_index=1,
          name='num',
          fill_null_func=lambda x, y, z: np.zeros((1,))
        )
        trans.calc_global_values(self.array)
        ci = trans.col_index

        for row_index, row in enumerate(self.array):
          arrays_dict = trans.forward_transform(row, verbose=False)
          val = trans.backward_transform(arrays_dict, verbose=False)
          filled_null = row[ci: ci + 1] if row_index != 1 else np.zeros((1,))
          th.assert_arrays_equal(self, arrays_dict['is_null'], np.isnan(row[ci: ci + 1]))
          th.assert_arrays_equal(self, arrays_dict['val'], filled_null)

          if row_index != 1:
            th.assert_arrays_equal(self, row[ci: ci + 1], val)
          else:
            th.assert_arrays_equal(self, np.isnan(row[ci: ci + 1]), np.isnan(val))

    def test_mean_std(self):
        trans = n.NumTransform(
          col_index=1,
          name='num',
          norm_mode='mean_std',
          fill_null_func=lambda x, y, z: np.zeros((1,))
        )
        trans.calc_global_values(self.array)
        ci = trans.col_index

        mean = trans.mean
        std = trans.std
        for row_index, row in enumerate(self.array):
          arrays_dict = trans.forward_transform(row, verbose=False)
          val = trans.backward_transform(arrays_dict, verbose=False)

          if row_index != 1:
            filled_null = (row[ci: ci + 1] - mean)/std
          else:
            filled_null = (np.zeros((1,)) - mean)/std

          th.assert_arrays_equal(self, arrays_dict['is_null'], np.isnan(row[ci: ci + 1]))
          th.assert_arrays_equal(self, arrays_dict['val'], filled_null)

          if row_index != 1:
            th.assert_arrays_equal(self, row[ci: ci + 1], val)
          else:
            th.assert_arrays_equal(self, np.isnan(row[ci: ci + 1]), np.isnan(val))

    def test_min_max(self):
        trans = n.NumTransform(
          col_index=1,
          name='num',
          norm_mode='min_max',
          fill_null_func=lambda x, y, z: np.zeros((1,))
        )
        trans.calc_global_values(self.array)
        ci = trans.col_index

        min = trans.min
        max = trans.max
        self.assertEqual((min, max), (2, 11))
        for row_index, row in enumerate(self.array):
          arrays_dict = trans.forward_transform(row, verbose=False)
          val = trans.backward_transform(arrays_dict, verbose=False)

          if row_index != 1:
            filled_null = (row[ci: ci + 1] - min)/(max - min)
          else:
            filled_null = (np.zeros((1,)) - min)/(max - min)

          th.assert_arrays_equal(self, arrays_dict['is_null'], np.isnan(row[ci: ci + 1]))
          th.assert_arrays_equal(self, arrays_dict['val'], filled_null)

          if row_index != 1:
            th.assert_arrays_equal(self, row[ci: ci + 1], val)
          else:
            th.assert_arrays_equal(self, np.isnan(row[ci: ci + 1]), np.isnan(val))

    def test_errors(self):
      with self.assertRaises(ValueError):
        trans = n.NumTransform(
          name='num'
        )

      with self.assertRaises(ValueError):
        trans = n.NumTransform(
          norm_mode='whatever',
          name='num'
        )

      trans = n.NumTransform(
        col_index=2,
        name='num',
        norm_mode='min_max',
      )

      with self.assertRaises(ValueError):
        trans.calc_global_values(self.array)

      trans = n.NumTransform(
        col_index=2,
        name='num',
        norm_mode='mean_std',
      )

      with self.assertRaises(ValueError):
        trans.calc_global_values(self.array)


if __name__ == "__main__":
    unittest.main()
