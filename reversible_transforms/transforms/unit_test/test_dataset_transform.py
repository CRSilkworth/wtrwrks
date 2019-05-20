import shutil
import tempfile
import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.transforms.dataset_transform as tr
import reversible_transforms.transforms.cat_transform as ct
import reversible_transforms.transforms.num_transform as nt
import reversible_transforms.transforms.datetime_transform as dt
import reversible_transforms.transforms.string_transform as st
import numpy as np
import datetime
import nltk

en_tokenizer = nltk.word_tokenize


class TestDatasetTransform(th.TestDataset):
  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir)

  def test_dataset_transform(self):
    array = self._get_array()

    dataset_transform = tr.DatasetTransform(name='DT')
    dataset_transform.add_transform(
      col_ranges=[0, 1],
      transform=ct.CatTransform(
        name='CAT',
        norm_mode='mean_std'
      )
    )
    dataset_transform.add_transform(
      col_ranges=[3, 6],
      transform=dt.DateTimeTransform(
        name='DATE',
        norm_mode='min_max',
        fill_nat_func=lambda a: np.array(datetime.datetime(1950, 1, 1)),
      )
    )
    dataset_transform.add_transform(
      col_ranges=[6, 9],
      transform=nt.NumTransform(
        name='NUM',
        norm_mode='min_max',

        fill_nan_func=lambda a: np.array(0),
      )
    )
    dataset_transform.add_transform(
      col_ranges=[9, 11],
      transform=st.StringTransform(
        name='STRING',
        tokenizer=en_tokenizer,
        index_to_word=['__UNK__'] + self._get_index_to_word(array[:, 9:11], en_tokenizer),
        unk_index=0,
      )
    )
    dataset_transform.calc_global_values(array)

    for i in xrange(2):
      self.pour_pump(
        dataset_transform,
        array,
        {
          'STRING': {'tokenizer': en_tokenizer}
        },
        {
          'DT/CAT/indices': [[0, 1, 2, 0]],
          'DT/CAT/missing_vals': np.array([]),
          'DT/CAT/one_hots': [[[1.0, -0.5773502691896258, -0.5773502691896258], [-1.0, 1.7320508075688774, -0.5773502691896258], [-1.0, -0.5773502691896258, 1.7320508075688774], [1.0, -0.5773502691896258, -0.5773502691896258]]],
          'DT/DATE/nats':  [[False, False, False, False], [False, True, False, False], [True, True, True, True]],
          'DT/DATE/diff': np.array([], dtype='timedelta64[us]'),
          'DT/DATE/nums': np.array([[0.0, 0.01694915254237288, 0.03389830508474576, 0.0], [0.0, -427.1525423728813, 0.5254237288135594, 1.0], [-427.1525423728813, -427.1525423728813, -427.1525423728813, -427.1525423728813]]),
          'DT/NUM/nans': [[False, False, False, False], [False, True, False, False], [True, True, True, True]],
          'DT/NUM/nums': [[0.0, 0.3, 0.6, 0.9], [0.1, -0.1, 0.7, 1.0], [-0.1, -0.1, -0.1, -0.1]],
          'DT/STRING/indices': [[[9, 29, 50, 30, 29, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [7, 16, 43, 28, 49, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [12, 41, 29, 34, 54, 2, 30, 1, 18, 3, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1], [4, 32, 33, 14, 48, 44, 31, 51, 47, 43, 52, 17, 3, -1, -1, -1, -1, -1, -1, -1]], [[6, 38, 2, 23, 49, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [5, 26, 53, 31, 22, 50, 8, 46, 42, 15, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1], [13, 21, 45, 39, 27, 14, 20, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [11, 1, 24, 19, 36, 43, 40, 35, 25, 37, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1]]],
          'DT/STRING/missing_vals': np.array([], dtype='|U59'),
          'DT/STRING/tokenize_diff': np.array([['[["d", 16, 17, ""], ["d", 18, 32, ""]]', '[["d", 24, 25, ""], ["d", 26, 40, ""]]', '[["d", 21, 22, ""], ["d", 26, 27, ""], ["d", 37, 38, ""], ["d", 42, 43, ""], ["d", 44, 52, ""]]', '[["d", 2, 3, ""], ["d", 57, 58, ""], ["d", 59, 66, ""]]'], ['[["d", 8, 9, ""], ["d", 19, 20, ""], ["d", 21, 35, ""]]', '[["d", 58, 59, ""], ["d", 60, 69, ""]]', '[["d", 35, 36, ""], ["d", 37, 49, ""]]', '[["d", 3, 4, ""], ["d", 45, 46, ""], ["d", 47, 56, ""]]']], dtype='|S95'),
          'DT/Partition_0/tubes/missing_cols': np.array([1, 2]),
          'DT/Partition_0/tubes/missing_array': np.array([['b', 'None', 'b', 'c'], [1.0, 2.0, np.nan, 1.0]], dtype=np.object),
        }
      )
      # dataset_transform = self.write_read(dataset_transform, self.temp_dir)

  #
  def test_read_write_transform(self):
    array = self._get_array()

    dataset_transform = tr.DatasetTransform(name='DT')
    dataset_transform.add_transform(
      col_ranges=[0, 1],
      transform=ct.CatTransform(
        name='CAT',
        norm_mode='mean_std'
      )
    )
    dataset_transform.add_transform(
      col_ranges=[3, 6],
      transform=dt.DateTimeTransform(
        name='DATE',
        norm_mode='min_max',
        fill_nat_func=lambda a: np.array(datetime.datetime(1950, 1, 1)),
      )
    )
    dataset_transform.add_transform(
      col_ranges=[6, 9],
      transform=nt.NumTransform(
        name='NUM',
        norm_mode='min_max',

        fill_nan_func=lambda a: np.array(0),
      )
    )
    dataset_transform.add_transform(
      col_ranges=[9, 11],
      transform=st.StringTransform(
        name='STRING',
        tokenizer=en_tokenizer,
        index_to_word=['__UNK__'] + self._get_index_to_word(array[:, 9:11], en_tokenizer),
        unk_index=0,
      )
    )
    dataset_transform.calc_global_values(array)

    for i in xrange(2):
      self.pour_pump(
        dataset_transform,
        array,
        {
          'STRING': {'tokenizer': en_tokenizer}
        },
        {
          'DT/CAT/indices': [[0, 1, 2, 0]],
          'DT/CAT/missing_vals': np.array([]),
          'DT/CAT/one_hots': [[[1.0, -0.5773502691896258, -0.5773502691896258], [-1.0, 1.7320508075688774, -0.5773502691896258], [-1.0, -0.5773502691896258, 1.7320508075688774], [1.0, -0.5773502691896258, -0.5773502691896258]]],
          'DT/DATE/nats':  [[False, False, False, False], [False, True, False, False], [True, True, True, True]],
          'DT/DATE/diff': np.array([], dtype='timedelta64[us]'),
          'DT/DATE/nums': np.array([[0.0, 0.01694915254237288, 0.03389830508474576, 0.0], [0.0, -427.1525423728813, 0.5254237288135594, 1.0], [-427.1525423728813, -427.1525423728813, -427.1525423728813, -427.1525423728813]]),
          'DT/NUM/nans': [[False, False, False, False], [False, True, False, False], [True, True, True, True]],
          'DT/NUM/nums': [[0.0, 0.3, 0.6, 0.9], [0.1, -0.1, 0.7, 1.0], [-0.1, -0.1, -0.1, -0.1]],
          'DT/STRING/indices': [[[9, 29, 50, 30, 29, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [7, 16, 43, 28, 49, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [12, 41, 29, 34, 54, 2, 30, 1, 18, 3, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1], [4, 32, 33, 14, 48, 44, 31, 51, 47, 43, 52, 17, 3, -1, -1, -1, -1, -1, -1, -1]], [[6, 38, 2, 23, 49, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [5, 26, 53, 31, 22, 50, 8, 46, 42, 15, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1], [13, 21, 45, 39, 27, 14, 20, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [11, 1, 24, 19, 36, 43, 40, 35, 25, 37, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1]]],
          'DT/STRING/missing_vals': np.array([], dtype='|U59'),
          'DT/STRING/tokenize_diff': np.array([['[["d", 16, 17, ""], ["d", 18, 32, ""]]', '[["d", 24, 25, ""], ["d", 26, 40, ""]]', '[["d", 21, 22, ""], ["d", 26, 27, ""], ["d", 37, 38, ""], ["d", 42, 43, ""], ["d", 44, 52, ""]]', '[["d", 2, 3, ""], ["d", 57, 58, ""], ["d", 59, 66, ""]]'], ['[["d", 8, 9, ""], ["d", 19, 20, ""], ["d", 21, 35, ""]]', '[["d", 58, 59, ""], ["d", 60, 69, ""]]', '[["d", 35, 36, ""], ["d", 37, 49, ""]]', '[["d", 3, 4, ""], ["d", 45, 46, ""], ["d", 47, 56, ""]]']], dtype='|S95'),
          'DT/Partition_0/tubes/missing_cols': np.array([1, 2]),
          'DT/Partition_0/tubes/missing_array': np.array([['b', 'None', 'b', 'c'], [1.0, 2.0, np.nan, 1.0]], dtype=np.object),
        }
      )
      self.write_read_example(dataset_transform, array, self.temp_dir)
      dataset_transform = self.write_read(dataset_transform, self.temp_dir)

  def _get_array(self):
    cat_array = np.array([
      ['a', 'b', 1.0],
      ['b', 'None', 2.0],
      ['c', 'b', np.nan],
      ['a', 'c', 1.0],
    ], dtype=np.object)

    datetime_array = np.array([
      ['2019-01-01', '2019-01-01', np.datetime64('NaT')],
      ['2019-01-02', np.datetime64('NaT'), np.datetime64('NaT')],
      ['2019-01-03', '2019-02-01', np.datetime64('NaT')],
      ['2019-01-01', '2019-03-01', np.datetime64('NaT')]
    ], dtype=np.datetime64)

    num_array = np.array([
      [1, 2, np.nan],
      [4, np.nan, np.nan],
      [7, 8, np.nan],
      [10, 11, np.nan]
    ])

    string_array = np.array([
        ["It is what it is.", "Get sick, get well."],
        ["Hang around the ink well.", "Everybody here would know exactly what I was talking about."],
        ["The sun is not yellow, it's chicken. OK.", "They ended up sleeping in a doorway."],
        ["Don't need a weatherman to know which way the wind blows.", "She's got diamonds on the soles of her shoes."]
    ])
    r_array = np.concatenate([cat_array, datetime_array, num_array, string_array], axis=1)

    return r_array

  def _get_index_to_word(self, strings, tokenizer, lemmatizer=None, half_width=False):
    index_to_word = set()
    for string in strings.flatten():
      tokens = tokenizer(string)
      for token in tokens:
        if half_width:
          token = _half_width(token)
        if lemmatizer is not None:
          token = lemmatizer(token)

        index_to_word.add(token)
    index_to_word = sorted(index_to_word)
    return index_to_word
if __name__ == "__main__":
  unittest.main()
