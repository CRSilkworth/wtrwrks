import transform_objects as o
import shutil
import tempfile
import unittest
import production.utils.test_helpers as th
import production.transforms as n
import production.transforms.transform_set as ns
import os
import pandas as pd
import numpy as np

class TestTransform(unittest.TestCase):
  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir)

  def test_set_transform(self):
    cat_norm = n.SetTransform(o.df, ['E'], verbose=False)
    for num, (index, row) in enumerate(o.df.iterrows()):
      vector = cat_norm.row_to_vector(row, verbose=False)

      th.assert_arrays_equal(self, vector, o.E_vectors[num], threshold=0.1)

      self.assertTrue(
        (cat_norm.vector_to_row(cat_norm.row_to_vector(row)) == row[['E']]).all()
      )

    path = os.path.join(self.temp_dir, 'cat_norm.pickle')
    cat_norm.save_to_file(path)

    cat_norm = n.SetTransform(from_file=path, verbose=False)

    for num, (index, row) in enumerate(o.df.iterrows()):
      vector = cat_norm.row_to_vector(row, verbose=False)
      th.assert_arrays_equal(self, vector, o.E_vectors[num], threshold=0.1)

      self.assertTrue(
        (cat_norm.vector_to_row(cat_norm.row_to_vector(row)) == row[['E']]).all()
      )
  def test_cat_transform(self):
    cat_norm = n.CatTransform(o.df, ['A'])
    for num, (index, row) in enumerate(o.df.iterrows()):
      th.assert_arrays_equal(self, cat_norm.row_to_vector(row, verbose=False), o.A_vectors[num], threshold=0.1)

      self.assertTrue(
        (cat_norm.vector_to_row(cat_norm.row_to_vector(row)) == row[['A']]).all()
      )

    row['A'] = 'k'
    th.assert_arrays_equal(self, cat_norm.row_to_vector(row, verbose=False), -cat_norm.means/cat_norm.stds)

    path = os.path.join(self.temp_dir, 'cat_norm.pickle')
    cat_norm.save_to_file(path)

    cat_norm = n.CatTransform(from_file=path)

    for num, (index, row) in enumerate(o.df.iterrows()):
      th.assert_arrays_equal(self, cat_norm.row_to_vector(row), o.A_vectors[num], threshold=0.1)

      self.assertTrue(
        (cat_norm.vector_to_row(cat_norm.row_to_vector(row)) == row[['A']]).all()
      )

  def test_num_transform(self):
    num_norm = n.NumTransform(o.df, ['B'])
    for num, (index, row) in enumerate(o.df.iterrows()):
      th.assert_arrays_equal(self, num_norm.row_to_vector(row), o.B_vectors[num], threshold=0.1)

      self.assertTrue(
        np.abs(
          num_norm.vector_to_row(
            num_norm.row_to_vector(row)
          ) - row[['B']]
          ).all() < 0.0001)

    path = os.path.join(self.temp_dir, 'num_norm.pickle')
    num_norm.save_to_file(path)

    num_norm = n.NumTransform(from_file=path)

    for num, (index, row) in enumerate(o.df.iterrows()):
      th.assert_arrays_equal(self, num_norm.row_to_vector(row), o.B_vectors[num], threshold=0.1)

      self.assertTrue(
        np.abs(
          num_norm.vector_to_row(
            num_norm.row_to_vector(row)
          ) - row[['B']]
          ).all() < 0.0001)

  def test_date_transform(self):
    num_norm = n.DateTimeTransform(o.df, ['D'], start_datetimes=[pd.Timestamp(2018, 1, 1)])
    for num, (index, row) in enumerate(o.df.iterrows()):
      # pass
      th.assert_arrays_equal(self, num_norm.row_to_vector(row), o.D_vectors[num], threshold=0.1)
      self.assertTrue(((num_norm.vector_to_row(num_norm.row_to_vector(row)) - row[['D']]) < pd.to_timedelta('1s')).all())

    path = os.path.join(self.temp_dir, 'num_norm.pickle')
    num_norm.save_to_file(path)

    num_norm = n.DateTimeTransform(from_file=path)

    for num, (index, row) in enumerate(o.df.iterrows()):
      th.assert_arrays_equal(self, num_norm.row_to_vector(row), o.D_vectors[num], threshold=0.1)
      self.assertTrue(((num_norm.vector_to_row(num_norm.row_to_vector(row)) - row[['D']]) < pd.to_timedelta('1s')).all())

  def test_time_series_transform(self):
    num_norm = n.TimeSeriesTransform(o.df, ['B', 'C'], groupby='A')
    for num, (index, row) in enumerate(o.df.iterrows()):
      th.assert_arrays_equal(self, num_norm.row_to_vector(row), o.BC_vectors[num], threshold=0.1)
      a = row['A']
      r_row = num_norm.vector_to_row(
        num_norm.row_to_vector(row),
        groupby_val=a
      )
      self.assertTrue(np.abs((r_row - row[['B', 'C']]) < 0.001).all())

    path = os.path.join(self.temp_dir, 'num_norm.pickle')
    num_norm.save_to_file(path)

    num_norm = n.TimeSeriesTransform(from_file=path)

    for num, (index, row) in enumerate(o.df.iterrows()):
      th.assert_arrays_equal(self, num_norm.row_to_vector(row), o.BC_vectors[num], threshold=0.1)
      a = row['A']
      r_row = num_norm.vector_to_row(
        num_norm.row_to_vector(row),
        groupby_val=a
      )
      self.assertTrue(np.abs((r_row - row[['B', 'C']]) < 0.001).all())

  def test_transform_set(self):
    norm_set = ns.TransformSet()
    norm_set.add_row_num_mappings(key='whatever', df=o.df, index_column='I')
    self.assertEqual(norm_set.index_to_row_num, {'whatever': {'1': 1, '0': 0, '3': 3, '2': 2, '5': 5, '4': 4}})
    self.assertEqual(norm_set.row_num_to_index, {'whatever': ['0', '1', '2', '3', '4', '5']})

    norm_set['A'] = n.CatTransform(o.df, ['A'], verbose=False)
    norm_set['B'] = n.NumTransform(o.df, ['B'], verbose=False)
    norm_set['D'] = n.DateTimeTransform(o.df, ['D'], start_datetimes=[pd.Timestamp(2018, 1, 1)], verbose=False)
    norm_set['BC'] = n.TimeSeriesTransform(o.df, ['B', 'C'], groupby='A', verbose=False)

    for num, (index, row) in enumerate(o.df.iterrows()):
      vector = norm_set.row_to_vector(row, verbose=False)
      th.assert_arrays_equal(self, vector, o.full_vectors[num])

      r_row = norm_set.vector_to_row(vector)
      for col in ['A', 'B', 'C', 'D']:
        if col in ('B', 'C'):
          self.assertTrue(
            (np.abs(row[col] - r_row[col]) < 0.01).all()
          )
        if col == 'A':
          self.assertTrue(row[col] == r_row[col])
        if col == 'D':
          self.assertTrue(
            (r_row[col] - row[col]) < pd.to_timedelta('1s')
          )
    path = os.path.join(self.temp_dir, 'norm_set.pickle')
    norm_set.save_to_file(path)

    norm_set = ns.TransformSet(from_file=path)
    self.assertEqual(norm_set.index_to_row_num, {'whatever': {'1': 1, '0': 0, '3': 3, '2': 2, '5': 5, '4': 4}})
    self.assertEqual(norm_set.row_num_to_index, {'whatever': ['0', '1', '2', '3', '4', '5']})
    for num, (index, row) in enumerate(o.df.iterrows()):
      vector = norm_set.row_to_vector(row, verbose=False)
      th.assert_arrays_equal(self, vector, o.full_vectors[num])

      r_row = norm_set.vector_to_row(vector)
      for col in ['A', 'B', 'C', 'D']:
        if col in ('B', 'C'):
          self.assertTrue(
            (np.abs(row[col] - r_row[col]) < 0.01).all()
          )
        if col == 'A':
          self.assertTrue(row[col] == r_row[col])
        if col == 'D':
          self.assertTrue(
            (r_row[col] - row[col]) < pd.to_timedelta('1s')
          )
if __name__ == "__main__":
  unittest.main()
