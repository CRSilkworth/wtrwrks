import transform_objects as o
import shutil
import tempfile
import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.transforms as tr
import os
import pandas as pd
import numpy as np
import datetime
import nltk

en_tokenizer = nltk.word_tokenize

class TestTransform(th.TestTransform):
  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir)

  def test_set_transform(self):
    array = self._get_array()

    dataset_transform = tr.DatasetTransform('DT')
    dataset_transform.add_transform(
      col_ranges=[0, 1],
      transform=tr.CatTransform(
        name='CAT',
        norm_mode='mean_std'
      )
    )
    dataset_transform.add_transform(
      col_ranges=[3, 6],
      transform=tr.DateTimeTransform(
        name='DATE',
        norm_mode='min_max',
        fill_nat_func=lambda a: np.array(datetime.datetime(1950, 1, 1)),
      )
    )
    dataset_transform.add_transform(
      col_ranges=[6, 9],
      transform=tr.NumTransform(
        name='NUM',
        norm_mode='min_max',
        fill_nan_func=lambda a: np.array(0),
      )
    )
    dataset_transform.add_transform(
      col_ranges=[9, 11],
      transform=tr.NumTransform(
        name='STRING',
        tokenizer=en_tokenizer,
        index_to_word=['__UNK__'] + self._get_index_to_word(array[9, 11], en_tokenizer),
        unk_index=0,
      )
    )
    dataset_transform.calc_global_values(array)

    for i in xrange(2):
      self.dataset_pour_pump(
        dataset_transform,
        array,
        {
          'STRING': {'tokenizer': en_tokenizer}
        },
        {
          'nats': [[False], [False], [False], [False]],
          'diff': np.array([], dtype='timedelta64[us]')
        }
      )
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

    return np.concatenate([cat_array, datetime_array, num_array, string_array], axis=1)

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
