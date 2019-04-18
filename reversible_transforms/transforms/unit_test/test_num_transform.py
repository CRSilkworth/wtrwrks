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
        for i in xrange(2):
          arrays_dict = trans.forward_transform(self.array, verbose=False)
          val = trans.backward_transform(arrays_dict, verbose=False)

          th.assert_arrays_equal(self, np.zeros((4, 1), dtype=np.bool), arrays_dict['isnan'])
          th.assert_arrays_equal(self, self.array[:, ci: ci + 1], arrays_dict['data'])
          th.assert_arrays_equal(self, self.array[:, ci: ci + 1], val)

          temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
          trans.save_to_file(temp_file_path)
          trans = n.NumTransform(from_file=temp_file_path)

    def test_nan(self):
      def fill(array, col_index):
        col = np.array(array[:, col_index: col_index + 1], copy=True)
        col[np.isnan(col)] = 0.0
        return col
      trans = n.NumTransform(
        col_index=1,
        name='num',
        fill_nan_func=fill
      )
      trans.calc_global_values(self.array)
      ci = trans.col_index

      for i in xrange(2):
        arrays_dict = trans.forward_transform(self.array, verbose=False)
        val = trans.backward_transform(arrays_dict, verbose=False)

        isnan = np.zeros((4, 1), dtype=np.bool)
        isnan[1] = True
        data = np.array(self.array[:, ci: ci + 1], copy=True)
        data = data[~isnan]
        th.assert_arrays_equal(self, isnan, arrays_dict['isnan'])
        th.assert_arrays_equal(self, data, arrays_dict['data'][~isnan])

        th.assert_arrays_equal(self, val, self.array[:, ci: ci + 1])

        temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
        trans.save_to_file(temp_file_path)
        trans = n.NumTransform(from_file=temp_file_path)

    def test_mean_std(self):
      def fill(array, col_index):
        col = np.array(array[:, col_index: col_index + 1], copy=True)
        col[np.isnan(col)] = 0.0
        return col

      trans = n.NumTransform(
        col_index=1,
        name='num',
        norm_mode='mean_std',
        fill_nan_func=fill
      )
      trans.calc_global_values(self.array)
      ci = trans.col_index
      for i in xrange(2):
        arrays_dict = trans.forward_transform(self.array, verbose=False)
        val = trans.backward_transform(arrays_dict, verbose=False)

        isnan = np.zeros((4, 1), dtype=np.bool)
        isnan[1] = True
        data = np.array(self.array[:, ci: ci + 1], copy=True)
        data = (data[~isnan] - trans.mean)/trans.std

        th.assert_arrays_equal(self, isnan, arrays_dict['isnan'])
        th.assert_arrays_equal(self, data, arrays_dict['data'][~isnan])
        th.assert_arrays_equal(self, val, self.array[:, ci: ci + 1])

        temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
        trans.save_to_file(temp_file_path)
        trans = n.NumTransform(from_file=temp_file_path)

    def test_min_max(self):
      def fill(array, col_index):
        col = np.array(array[:, col_index: col_index + 1], copy=True)
        col[np.isnan(col)] = 0.0
        return col
      trans = n.NumTransform(
        col_index=1,
        name='num',
        norm_mode='min_max',
        fill_nan_func=fill
      )
      trans.calc_global_values(self.array)
      ci = trans.col_index
      for i in xrange(2):
        arrays_dict = trans.forward_transform(self.array, verbose=False)
        val = trans.backward_transform(arrays_dict, verbose=False)

        isnan = np.zeros((4, 1), dtype=np.bool)
        isnan[1] = True
        data = np.array(self.array[:, ci: ci + 1], copy=True)
        data = (data[~isnan] - trans.min)/(trans.max - trans.min)

        th.assert_arrays_equal(self, isnan, arrays_dict['isnan'])
        th.assert_arrays_equal(self, data, arrays_dict['data'][~isnan])
        th.assert_arrays_equal(self, val, self.array[:, ci: ci + 1])

        temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
        trans.save_to_file(temp_file_path)
        trans = n.NumTransform(from_file=temp_file_path)

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
