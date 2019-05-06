import shutil
import tempfile
import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.transforms.datetime_transform as n
import os
import pandas as pd
import numpy as np
import datetime


class TestDateTimeTransform(th.TestTransform):
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
        name='datetime'
      )
      trans.calc_global_values(self.array[:, 0: 1])
      target = np.array((self.array[:, 0: 1] - trans.zero_datetime) / np.timedelta64(1, 'D'), copy=True)
      for i in xrange(2):
        self.pour_pump(
          trans,
          self.array[:, 0: 1],
          {
            'nums': target,
            # 'replaced_vals': np.array([], dtype='datetime64[D]'),
            'nats': [[False], [False], [False], [False]],
            'diff': np.array([], dtype='timedelta64[us]')
          }
        )
        trans = self.write_read(trans, self.temp_dir)

    def test_nan(self):
      def fill(array):
        return np.array(datetime.datetime(1970, 1, 1))
      trans = n.DateTimeTransform(
        name='datetime',
        fill_nat_func=fill,
        time_unit='W',
        num_units=2,
        zero_datetime=datetime.datetime(2000, 1, 1)
      )
      trans.calc_global_values(self.array[:, 1: 2])
      target = np.array((self.array[:, 1: 2] - trans.zero_datetime) / np.timedelta64(2, 'W'), copy=True)
      target[np.isnan(target)] = -782.64285714
      for i in xrange(2):
        self.pour_pump(
          trans,
          self.array[:, 1: 2],
          {
            'nums': target,
            'nats': [[False], [True], [False], [False]],
            'diff': np.array([[259200000000], [-172800000000], [518400000000], [518400000000]], dtype='timedelta64[us]')
          }
        )
        trans = self.write_read(trans, self.temp_dir)

    def test_mean_std(self):

      def fill(array):
        unique = np.unique(array[~np.isnat(array)])
        replace_with = np.full(array[np.isnat(array)].shape, unique[0])
        return replace_with

      trans = n.DateTimeTransform(
        norm_mode='mean_std',
        fill_nat_func=fill
      )
      trans.calc_global_values(self.array[:, 1: 2])
      target = np.array((self.array[:, 1: 2] - trans.zero_datetime) / np.timedelta64(1, 'D'), copy=True)
      target = (target - trans.mean)/trans.std
      target[1, 0] = -1.24496691
      for i in xrange(2):
        self.pour_pump(
          trans,
          self.array[:, 1: 2],
          {
            'nums': target,
            'nats': [[False], [True], [False], [False]],
            'diff': np.array([], dtype='timedelta64[us]')
          }
        )
        trans = self.write_read(trans, self.temp_dir)

    def test_min_max(self):
      def fill(array):
        mins = np.expand_dims(np.min(array, axis=0), axis=0)
        mins = np.tile(mins, reps=[4, 1])
        replace_with = mins[np.isnat(array)]
        return replace_with
      trans = n.DateTimeTransform(
        fill_nat_func=fill,
        norm_mode='min_max'
      )
      trans.calc_global_values(self.array[:, 0: 1])
      target = np.array((self.array[:, 0: 1] - trans.zero_datetime) / np.timedelta64(1, 'D'), copy=True)
      target = (target - trans.min)/(trans.max - trans.min)

      for i in xrange(2):
        self.pour_pump(
          trans,
          self.array[:, 0: 1],
          {
            'nums': target,
            'nats': [[False], [False], [False], [False]],
            'diff': np.array([], dtype='timedelta64[us]')
          }
        )
        trans = self.write_read(trans, self.temp_dir)

    def test_errors(self):

      with self.assertRaises(ValueError):
        trans = n.DateTimeTransform(
          norm_mode='whatever',
          name='datetime'
        )

      with self.assertRaises(ValueError):
        trans = n.DateTimeTransform(
          name='datetime',
          norm_mode='whatever',
        )

      trans = n.DateTimeTransform(
        name='datetime',
        norm_mode='mean_std',
      )
      with self.assertRaises(AssertionError):
        trans.get_waterwork()


if __name__ == "__main__":
    unittest.main()
