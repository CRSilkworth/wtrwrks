import shutil
import tempfile
import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.transforms.cat_transform as n
import os
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

    # def test_no_norm(self):
    #   trans = n.CatTransform(
    #     name='cat'
    #   )
    #   trans.calc_global_values(self.array[:, 0: 1])
    #   target = np.array([
    #     [1., 0., 0.],
    #     [0., 1., 0.],
    #     [0., 0., 1.],
    #     [1., 0., 0.]
    #   ])
    #
    #   for i in xrange(2):
    #     self.pour_pump(
    #       trans,
    #       self.array[:, 0: 1],
    #       {
    #         'missing_vals': [],
    #         'one_hots': target,
    #       }
    #     )
    #     trans = self.write_read(trans, self.temp_dir)
    #
    # def test_valid_cats(self):
    #   trans = n.CatTransform(
    #     name='cat',
    #     valid_cats=['a', 'b']
    #   )
    #   trans.calc_global_values(self.array[:, 0: 1])
    #   target = np.array([
    #         [1, 0],
    #         [0, 1],
    #         [0, 0],
    #         [1, 0]
    #       ]).astype(float)
    #
    #   for i in xrange(2):
    #     self.pour_pump(
    #       trans,
    #       self.array[:, 0: 1],
    #       {
    #         'missing_vals': ['c'],
    #         'one_hots': target,
    #       }
    #     )
    #     trans = self.write_read(trans, self.temp_dir)
    #
    # def test_null(self):
    #   trans = n.CatTransform(
    #     name='cat',
    #   )
    #   trans.calc_global_values(self.array[:, 2: 3])
    #   target = np.array([
    #         [0, 1, 0],
    #         [0, 0, 1],
    #         [1, 0, 0],
    #         [0, 1, 0]
    #       ]).astype(float)
    #
    #   for i in xrange(2):
    #     self.pour_pump(
    #       trans,
    #       self.array[:, 2: 3],
    #       {
    #         'missing_vals': [],
    #         'one_hots': target,
    #       }
    #     )
    #     trans = self.write_read(trans, self.temp_dir)
    #
    # def test_ignore_null(self):
    #   trans = n.CatTransform(
    #     name='cat',
    #     ignore_null=True
    #   )
    #   trans.calc_global_values(self.array[:, 2: 3])
    #   target = np.array([
    #     [1, 0],
    #     [0, 1],
    #     [0, 0],
    #     [1, 0]
    #   ]).astype(float)
    #
    #   for i in xrange(2):
    #     self.pour_pump(
    #       trans,
    #       self.array[:, 2: 3],
    #       {
    #         'missing_vals': [np.nan],
    #         'one_hots': target,
    #       }
    #     )
    #     trans = self.write_read(trans, self.temp_dir)

    def test_mean_std(self):
      trans = n.CatTransform(
        name='cat',
        norm_mode='mean_std'
      )
      trans.calc_global_values(self.array[:, 1: 2])
      target = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
          ]).astype(float)
      target = (target - trans.mean)/trans.std
      for i in xrange(2):
        self.pour_pump(
          trans,
          self.array[:, 1: 2],
          {
            'missing_vals': [],
            'one_hots': target,
          }
        )
        trans = self.write_read(trans, self.temp_dir)
    #   trans.calc_global_values(self.array)
    #   for i in xrange(2):
    #     arrays_dict = trans.forward_transform(self.array, verbose=False)
    #     val = trans.backward_transform(arrays_dict, verbose=False)
    #
    #     data = np.array([
    #       [0, 1, 0],
    #       [1, 0, 0],
    #       [0, 1, 0],
    #       [0, 0, 1],
    #     ])
    #     data = (data - trans.mean)/trans.std
    #     index = np.expand_dims(np.array([1, 0, 1, 2]), axis=1)
    #     th.assert_arrays_equal(self, data, arrays_dict['data'])
    #     th.assert_arrays_equal(self, index, arrays_dict['index'])
    #     th.assert_arrays_equal(self, self.array[:, 1: 2], arrays_dict['cat_val'])
    #     th.assert_arrays_equal(self, self.array[:, 1: 2], val)
    #
    #     temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
    #     trans.save_to_file(temp_file_path)
    #     trans = n.CatTransform(from_file=temp_file_path)
    #
    # def test_errors(self):
    #   with self.assertRaises(ValueError):
    #     trans = n.CatTransform(
    #       name='cat'
    #     )
    #
    #   with self.assertRaises(ValueError):
    #     trans = n.CatTransform(
    #       norm_mode='whatever',
    #       name='cat'
    #     )
    #
    #   with self.assertRaises(ValueError):
    #     trans = n.CatTransform(
    #       norm_mode='min_max',
    #       name='cat'
    #     )
    #
    #   trans = n.CatTransform(
    #     name='cat',
    #     norm_mode='mean_std',
    #   )
    #   with self.assertRaises(AssertionError):
    #     trans.forward_transform(np.array([1]))
    #     with self.assertRaises(AssertionError):
    #       trans.forward_transform({})
    #
    #   with self.assertRaises(ValueError):
    #     trans.calc_global_values(np.array([[np.nan]]))

if __name__ == "__main__":
    unittest.main()
