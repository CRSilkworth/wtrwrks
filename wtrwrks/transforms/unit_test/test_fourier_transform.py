import shutil
import tempfile
import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.transforms.fourier_transform as n
import os
import pandas as pd
import numpy as np
import datetime


class TestFourierTransform(th.TestTransform):
    def setUp(self):
      self.temp_dir = tempfile.mkdtemp()
      self.times = np.array([
        ['2019-01-01'],
        [np.datetime64('NaT')],
        ['2019-02-01'],
        ['2019-03-01']
      ], dtype=np.dtype('datetime64[us]'))

      self.amps = np.array([
        [1.3],
        [np.nan],
        [1.7],
        [2.0]
      ])
      self.array = np.concatenate([self.times.astype(np.dtype('O')), self.amps.astype(np.dtype('O'))], axis=1)

    def tearDown(self):
      shutil.rmtree(self.temp_dir)

    def test_no_norm(self):
      def fill(array):
        return np.array(datetime.datetime(1970, 1, 1))

      zero_datetime = datetime.datetime(1970, 1, 1)
      end_datetime = np.max(self.times)
      trans = n.FourierTransform(
        name='datetime',
        zero_datetime=zero_datetime,
        end_datetime=end_datetime,
        num_frequencies=3
      )
      trans.calc_global_values(self.array)
      target = np.array([
        [0.0, 0.9967141902428158, 0.9934283804856316], [0.0, 0.0, 0.0], [0.0, 0.9984406326576075, 0.996881265315215], [0.0, 0.0, 0.0]
      ])
      div = np.array([
        [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 2.0]
      ])
      for i in xrange(2):
        self.pour_pump(
          trans,
          self.array,
          {
            'datetime/nums': target,
            'datetime/div': div,
            'datetime/amps': self.amps,
            'datetime/nats': [False, True, False, False],
            'datetime/diff': np.array([datetime.timedelta(0), datetime.timedelta(0), datetime.timedelta(0), datetime.timedelta(0)], dtype='timedelta64[us]')
          },
          test_type=False
        )
        trans = self.write_read(trans, self.temp_dir)

    # def test_df(self):
    #   def fill(array):
    #     mins = np.expand_dims(np.min(array, axis=0), axis=0)
    #     mins = np.tile(mins, reps=[array.shape[0], 1])
    #     replace_with = mins[np.isnat(array)]
    #     return replace_with
    #   trans = n.FourierTransform(
    #     name='datetime',
    #     fill_nat_func=fill,
    #     norm_mode='min_max'
    #   )
    #   df = pd.DataFrame({'a': self.array[0]})
    #   # trans.calc_global_values(self.array[:, 0: 1])
    #   trans.calc_global_values(data=df)
    #   target = np.array((self.array[:, 0: 1] - trans.zero_datetime) / np.timedelta64(1, 'D'), copy=True)
    #   target = (target - trans.min)/(trans.max - trans.min)
    #
    #   for i in xrange(2):
    #     self.pour_pump(
    #       trans,
    #       self.array[:, 0: 1],
    #       {
    #         'datetime/nums': target,
    #         'datetime/nats': [[False], [False], [False], [False]],
    #         'datetime/diff': np.array([[datetime.timedelta(0)], [datetime.timedelta(0)], [datetime.timedelta(0)], [datetime.timedelta(0)]], dtype='timedelta64[us]')
    #       },
    #       test_type=False
    #
    #     )
    #     trans = self.write_read(trans, self.temp_dir)
    #
    # def test_array_iter(self):
    #   def fill(array):
    #     mins = np.expand_dims(np.min(array, axis=0), axis=0)
    #     mins = np.tile(mins, reps=[array.shape[0], 1])
    #     replace_with = mins[np.isnat(array)]
    #     return replace_with
    #   trans = n.FourierTransform(
    #     name='datetime',
    #     fill_nat_func=fill,
    #     norm_mode='min_max'
    #   )
    #   array_iter = [self.array[0: 2, 0: 1], self.array[2: 4, 0: 1]]
    #   trans.calc_global_values(data_iter=array_iter)
    #   # trans.calc_global_values(self.array[:, 0: 1])
    #   target = np.array((self.array[:, 0: 1] - trans.zero_datetime) / np.timedelta64(1, 'D'), copy=True)
    #   target = (target - trans.min)/(trans.max - trans.min)
    #
    #   for i in xrange(2):
    #     self.pour_pump(
    #       trans,
    #       self.array[:, 0: 1],
    #       {
    #         'datetime/nums': target,
    #         'datetime/nats': [[False], [False], [False], [False]],
    #         'datetime/diff': np.array([[datetime.timedelta(0)], [datetime.timedelta(0)], [datetime.timedelta(0)], [datetime.timedelta(0)]], dtype='timedelta64[us]')
    #       },
    #       test_type=False
    #
    #     )
    #     trans = self.write_read(trans, self.temp_dir)
    #
    # def test_read_write(self):
    #   def fill(array):
    #     mins = np.expand_dims(np.min(array, axis=0), axis=0)
    #     mins = np.tile(mins, reps=[4, 1])
    #     replace_with = mins[np.isnat(array)]
    #     return replace_with
    #   trans = n.FourierTransform(
    #     name='datetime',
    #     fill_nat_func=fill,
    #     norm_mode='min_max'
    #   )
    #   trans.calc_global_values(self.array[:, 0: 1])
    #   target = np.array((self.array[:, 0: 1] - trans.zero_datetime) / np.timedelta64(1, 'D'), copy=True)
    #   target = (target - trans.min)/(trans.max - trans.min)
    #
    #   for i in xrange(2):
    #     self.pour_pump(
    #       trans,
    #       self.array[:, 0: 1],
    #       {
    #         'datetime/nums': target,
    #         'datetime/nats': [[False], [False], [False], [False]],
    #         'datetime/diff': np.array([[datetime.timedelta(0)], [datetime.timedelta(0)], [datetime.timedelta(0)], [datetime.timedelta(0)]], dtype='timedelta64[us]')
    #       },
    #       test_type=False
    #     )
    #     self.write_read_example(trans, self.array[:, 0: 1].astype(np.datetime64), self.temp_dir, test_type=False)
    #     trans = self.write_read(trans, self.temp_dir)

    def test_errors(self):

      with self.assertRaises(TypeError):
        trans = n.FourierTransform(
          name='datetime',
          zero_datetime=datetime.datetime(1970, 1, 1)
        )

      trans = n.FourierTransform(
        name='datetime',
        zero_datetime=datetime.datetime(1970, 1, 1),
        end_datetime=datetime.datetime(1970, 1, 1),
        num_frequencies=10,
      )
      with self.assertRaises(AssertionError):
        trans.get_waterwork()


if __name__ == "__main__":
    unittest.main()
