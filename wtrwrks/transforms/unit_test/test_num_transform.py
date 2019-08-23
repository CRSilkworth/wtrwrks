import shutil
import tempfile
import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.transforms.num_transform as n
import os
import pandas as pd
import numpy as np

class TestNumTransform(th.TestTransform):
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
          name='num'
        )
        trans.calc_global_values(self.array[:, 0: 1])
        target = self.array[:, 0: 1]
        for i in xrange(2):
          self.pour_pump(
            trans,
            self.array[:, 0: 1],
            {
              'num/nums': target,
              'num/nans': [[False], [False], [False], [False]],
            }
          )
          trans = self.write_read(trans, self.temp_dir)

    def test_nan(self):
      def fill(array):
        mins = np.expand_dims(np.nanmin(array, axis=0), axis=0)
        mins = np.tile(mins, reps=[4, 1])
        replace_with = mins[np.isnan(array)]
        return replace_with
      trans = n.NumTransform(
        name='num',
        fill_nan_func=fill
      )
      trans.calc_global_values(self.array[:, 1: 2])
      target = np.array(self.array[:, 1: 2], copy=True)
      target[1, 0] = 2
      for i in xrange(2):
        self.pour_pump(
          trans,
          self.array[:, 1: 2],
          {
            'num/nums': target,
            'num/nans': [[False], [True], [False], [False]],
          }
        )
        trans = self.write_read(trans, self.temp_dir)

    def test_mean_std(self):
      def fill(array):
        return np.array(0.)

      trans = n.NumTransform(
        name='num',
        norm_mode='mean_std',
        fill_nan_func=fill
      )
      trans.calc_global_values(self.array[:, 0: 1])
      target = self.array[:, 0: 1]
      target = (target - trans.mean)/(trans.std)
      for i in xrange(2):
        self.pour_pump(
          trans,
          self.array[:, 0: 1],
          {
            'num/nums': target,
            'num/nans': [[False], [False], [False], [False]],
          }
        )
        trans = self.write_read(trans, self.temp_dir)

    def test_min_max(self):
      def fill(array):
        return np.array(0.0)
      trans = n.NumTransform(
        name='num',
        norm_mode='min_max',
        fill_nan_func=fill,
        norm_axis=0
      )
      trans.calc_global_values(self.array[:, 0: 2])
      target = self.array[:, 0: 2]
      target = (target - trans.min)/(trans.max - trans.min)
      target[1, 1] = -trans.min[1]/(trans.max[1] - trans.min[1])
      for i in xrange(2):
        self.pour_pump(
          trans,
          self.array[:, 0: 2],
          {
            'num/nums': target,
            'num/nans': [[False, False], [False, True], [False, False], [False, False]],
          }
        )
        trans = self.write_read(trans, self.temp_dir)

    def test_df(self):
      def fill(array):
        return np.array(0.)

      trans = n.NumTransform(
        name='num',
        norm_mode='mean_std',
        fill_nan_func=fill
      )
      df = pd.DataFrame({'a': self.array[0]})
      # trans.calc_global_values(self.array[:, 0: 1])
      trans.calc_global_values(data=df)
      target = self.array[:, 0: 1]
      target = (target - trans.mean)/(trans.std)
      for i in xrange(2):
        self.pour_pump(
          trans,
          self.array[:, 0: 1],
          {
            'num/nums': target,
            'num/nans': [[False], [False], [False], [False]],
          }
        )
        trans = self.write_read(trans, self.temp_dir)

    def test_array_iter(self):
      def fill(array):
        return np.array(0.0)
      trans = n.NumTransform(
        name='num',
        norm_mode='min_max',
        fill_nan_func=fill,
        norm_axis=0
      )
      array_iter = [self.array[0: 2, 0: 2], self.array[2: 4, 0: 2]]
      trans.calc_global_values(data_iter=array_iter)
      # trans.calc_global_values(self.array[:, 0: 2])
      target = self.array[:, 0: 2]
      target = (target - trans.min)/(trans.max - trans.min)
      target[1, 1] = -trans.min[1]/(trans.max[1] - trans.min[1])
      for i in xrange(2):
        self.pour_pump(
          trans,
          self.array[:, 0: 2],
          {
            'num/nums': target,
            'num/nans': [[False, False], [False, True], [False, False], [False, False]],
          }
        )
        trans = self.write_read(trans, self.temp_dir)

    def test_read_write(self):
      def fill(array):
        return np.array(0.0)
      trans = n.NumTransform(
        name='NUM',
        norm_mode='min_max',

        fill_nan_func=lambda a: np.array(0),
      )
      trans.calc_global_values(self.array)
      for i in xrange(2):
        self.write_read_example(trans, self.array, self.temp_dir, test_type=False)

    def test_errors(self):

      with self.assertRaises(ValueError):
        trans = n.NumTransform(
          norm_mode='whatever',
          name='num'
        )

      trans = n.NumTransform(
        name='num',
        norm_mode='min_max',
      )

      with self.assertRaises(AssertionError):
        trans.get_waterwork()



if __name__ == "__main__":
    unittest.main()
