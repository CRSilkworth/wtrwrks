import shutil
import tempfile
import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.transforms.datetime_transform as n
import os
import pandas as pd
import numpy as np
import datetime


class TestDateTimeTransform(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.array = np.array([
          ['2019-01-01', '2019-01-01', np.datetime64('NaT')],
          ['2019-01-02', np.datetime64('NaT'), np.datetime64('NaT')],
          ['2019-01-03', '2019-02-01', np.datetime64('NaT')],
          ['2019-01-01', '2019-03-01', np.datetime64('NaT')]
        ], dtype=np.datetime64)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_no_norm(self):
        trans = n.DateTimeTransform(
          col_index=0,
          name='datetime'
        )
        trans.calc_global_values(self.array)
        ci = trans.col_index
        for i in xrange(2):
          arrays_dict = trans.forward_transform(self.array, verbose=False)
          val = trans.backward_transform(arrays_dict, verbose=False)

          th.assert_arrays_equal(self, np.zeros((4, 1), dtype=np.bool), arrays_dict['isnan'])
          th.assert_arrays_equal(self, (self.array[:, ci: ci + 1] - trans.zero_datetime) / np.timedelta64(1, "s"), arrays_dict['data'])
          th.assert_arrays_equal(self, self.array[:, ci: ci + 1], val)

          temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
          trans.save_to_file(temp_file_path)
          trans = n.DateTimeTransform(from_file=temp_file_path)
    def test_nan(self):
      def fill(array, col_index):
        col = np.array(array[:, col_index: col_index + 1], copy=True)
        col[np.isnat(col)] = datetime.datetime(1970, 1, 1)
        return col
      trans = n.DateTimeTransform(
        col_index=1,
        name='datetime',
        fill_nan_func=fill
      )
      trans.calc_global_values(self.array)
      ci = trans.col_index
      for i in xrange(2):
        arrays_dict = trans.forward_transform(self.array, verbose=False)
        val = trans.backward_transform(arrays_dict, verbose=False)

        isnan = np.zeros((4, 1), dtype=np.bool)
        isnan[1] = True
        data = np.array((self.array[:, ci: ci + 1] - trans.zero_datetime) / np.timedelta64(1, 's'), copy=True)
        data = data[~isnan]
        th.assert_arrays_equal(self, isnan, arrays_dict['isnan'])
        th.assert_arrays_equal(self, data, arrays_dict['data'][~isnan])

        th.assert_arrays_equal(self, val, self.array[:, ci: ci + 1])

        temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
        trans.save_to_file(temp_file_path)
        trans = n.DateTimeTransform(from_file=temp_file_path)

    def test_mean_std(self):
      def fill(array, col_index):
        col = np.array(array[:, col_index: col_index + 1], copy=True)
        col[np.isnat(col)] = datetime.datetime(1970, 1, 1)
        return col

      trans = n.DateTimeTransform(
        col_index=1,
        name='datetime',
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
        data = np.array((self.array[:, ci: ci + 1] - trans.zero_datetime) / np.timedelta64(1, 's'), copy=True)
        data = (data[~isnan] - trans.mean)/trans.std

        th.assert_arrays_equal(self, isnan, arrays_dict['isnan'])
        th.assert_arrays_equal(self, data, arrays_dict['data'][~isnan])
        th.assert_arrays_equal(self, val, self.array[:, ci: ci + 1])

        temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
        trans.save_to_file(temp_file_path)
        trans = n.DateTimeTransform(from_file=temp_file_path)

    def test_min_max(self):
      def fill(array, col_index):
        col = np.array(array[:, col_index: col_index + 1], copy=True)
        col[np.isnat(col)] = datetime.datetime(1970, 1, 1)
        return col
      trans = n.DateTimeTransform(
        col_index=1,
        name='datetime',
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
        data = np.array((self.array[:, ci: ci + 1] - trans.zero_datetime) / np.timedelta64(1, 's'), copy=True)
        data = (data[~isnan] - trans.min)/(trans.max - trans.min)

        th.assert_arrays_equal(self, isnan, arrays_dict['isnan'])
        th.assert_arrays_equal(self, data, arrays_dict['data'][~isnan])
        th.assert_arrays_equal(self, val, self.array[:, ci: ci + 1])

        temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
        trans.save_to_file(temp_file_path)
        trans = n.DateTimeTransform(from_file=temp_file_path)

    def test_errors(self):
      with self.assertRaises(ValueError):
        trans = n.DateTimeTransform(
          name='datetime'
        )

      with self.assertRaises(ValueError):
        trans = n.DateTimeTransform(
          norm_mode='whatever',
          name='datetime'
        )

      trans = n.DateTimeTransform(
        col_index=2,
        name='datetime',
        norm_mode='min_max',
      )

      with self.assertRaises(ValueError):
        trans.calc_global_values(self.array)

      trans = n.DateTimeTransform(
        col_index=2,
        name='datetime',
        norm_mode='mean_std',
      )

      with self.assertRaises(ValueError):
        trans.calc_global_values(self.array)


if __name__ == "__main__":
    unittest.main()
