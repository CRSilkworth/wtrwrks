# -*- coding: utf-8 -*-
import shutil
import tempfile
import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.transforms as n
import os
import pandas as pd
import numpy as np

class TestStringTransform(unittest.TestCase):
  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir)

  def test_en_normalize(self):
    indices = np.array([
      # [5, 11, 18, 12, 11, 3, -1, -1, -1, -1],
      # [8, 2, 4, 1, 13, 10, 15, 17, 19, 3],
      # [7, 16, 11, 14, 20, 2, 12, 1, 9, 3]
      [5, 10, 17, 11, 10, 3, -1, -1, -1, -1],
      [7, 2, 4, 1, 12, 9, 14, 16, 18, 3],
      [6, 15, 10, 13, 19, 2, 11, 1, 8, 3]
    ])

    strings = np.array([
      ["It is what it is."],
      ["Whatever, Bob's mother has seen the world."],
      ["The sun is not yellow, it's chicken. OK."]
    ])
    norm = n.StringTransform(
      col_index=0,
      language='en',
      max_vocab_size=20,
      max_sent_len=10
    )
    norm.calc_global_values(strings)
    for i in xrange(2):
      array_dict = norm.forward_transform(strings)
      th.assert_arrays_equal(self, array_dict['data'], indices, threshold=0.1)

      out_array = norm.backward_transform(array_dict)
      th.assert_arrays_equal(self, out_array, strings)

      temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
      norm.save_to_file(temp_file_path)
      norm = n.StringTransform(from_file=temp_file_path)

  def test_en_normalize_2(self):
    indices = np.array([
      [3, -1, -1, -1, -1, -1, -1, -1, -1, -1],
      [10, 2, 4, 1, 6, 8, 11, 3, -1, -1],
      [9, 12, 2, 1, 5, 3, 7, 3, -1, -1]
    ])
    strings = np.array([
      ["It is what it is."],
      ["Whatever, Bob's mother has seen the world."],
      ["The sun is not yellow, it's chicken. OK."]
    ])
    norm = n.StringTransform(
      col_index=0,
      language='en',
      max_vocab_size=20,
      max_sent_len=10,
      normalizer_kwargs={'lowercase': True, 'lemmatize': True, 'remove_stopwords': True}
    )
    norm.calc_global_values(strings)
    for i in xrange(2):
      array_dict = norm.forward_transform(strings)
      th.assert_arrays_equal(self, array_dict['data'], indices, threshold=0.1)

      out_array = norm.backward_transform(array_dict)
      th.assert_arrays_equal(self, out_array, strings)

      temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
      norm.save_to_file(temp_file_path)
      norm = n.StringTransform(from_file=temp_file_path)

  def test_ja_normalize(self):
    indices = np.array([
      # [13, 14, 17, 18, 23, 8, 22, 20, 9, 15, 16, 3, 21, -1, -1],
      # [6, 12, 4, 1, 11, 15, 19, 7, 5, 10, 2, -1, -1, -1, -1]
      [10, 0, 13, 14, 19, 0, 18, 16, 0, 11, 12, 3, 17, -1, -1],
      [0, 9, 4, 1, 8, 11, 15, 6, 5, 7, 2, -1, -1, -1, -1]
    ])
    strings = np.array([
      [u'チラシ・勧誘印刷物の無断投函は一切お断り'],
      [u'すみませんが、もう一度どお願いします。']
    ])
    norm = n.StringTransform(
      col_index=0,
      language='ja',
      max_vocab_size=20,
      max_sent_len=15
    )
    norm.calc_global_values(strings)
    for i in xrange(2):
      array_dict = norm.forward_transform(strings)
      th.assert_arrays_equal(self, array_dict['data'], indices, threshold=0.1)

      out_array = norm.backward_transform(array_dict)
      th.assert_arrays_equal(self, out_array, strings)

      temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
      norm.save_to_file(temp_file_path)
      norm = n.StringTransform(from_file=temp_file_path)

  def test_ja_normalize_2(self):
    indices = np.array([
      [10, 11, 0, 14, 19, 6, 18, 16, 7, 12, 13, 1, 17, -1, -1],
      [4, 9, 2, 0, 12, 15, 5, 3, 8, -1, -1, -1, -1, -1, -1]
    ])

    strings = np.array([
      [u'チラシ・勧誘印刷物の無断投函は一切お断り'],
      [u'すみませんが、もう一度どお願いします。']
    ])
    norm = n.StringTransform(
      col_index=0,
      language='ja',
      max_vocab_size=20,
      max_sent_len=15,
      normalizer_kwargs={'half_width': True, 'remove_stopwords': True}
    )
    norm.calc_global_values(strings)
    for i in xrange(2):
      array_dict = norm.forward_transform(strings)
      th.assert_arrays_equal(self, array_dict['data'], indices, threshold=0.1)

      out_array = norm.backward_transform(array_dict)
      th.assert_arrays_equal(self, out_array, strings)

      temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
      norm.save_to_file(temp_file_path)
      norm = n.StringTransform(from_file=temp_file_path)

  def test_zh_hans_normalize(self):
    indices = np.array([
      [11, 4, 1, 5, 8, 6, 3, -1, -1, -1, -1, -1, -1, -1, -1],
      [10, 9, 1, 5, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ])
    strings = np.array([
      [u'早上好,你好吗。'],
      [u'我很好,你呢?']
    ])
    norm = n.StringTransform(
      col_index=0,
      language='zh_hans',
      max_vocab_size=20,
      max_sent_len=15,
    )
    norm.calc_global_values(strings)
    for i in xrange(2):
      array_dict = norm.forward_transform(strings)
      th.assert_arrays_equal(self, array_dict['data'], indices, threshold=0.1)

      out_array = norm.backward_transform(array_dict)
      th.assert_arrays_equal(self, out_array, strings)

      temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
      norm.save_to_file(temp_file_path)
      norm = n.StringTransform(from_file=temp_file_path)

  def test_zh_hans_normalize_2(self):
    indices = np.array([
      [5, 3, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
      [4, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ])
    strings = np.array([
      [u'早上好,你好吗。'],
      [u'我很好,你呢?']
    ])
    norm = n.StringTransform(
      col_index=0,
      language='zh_hans',
      max_vocab_size=20,
      max_sent_len=15,
      normalizer_kwargs={'half_width': True, 'remove_stopwords': True}
    )
    norm.calc_global_values(strings)
    for i in xrange(2):
      array_dict = norm.forward_transform(strings)
      th.assert_arrays_equal(self, array_dict['data'], indices, threshold=0.1)

      out_array = norm.backward_transform(array_dict)
      th.assert_arrays_equal(self, out_array, strings)

      temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
      norm.save_to_file(temp_file_path)
      norm = n.StringTransform(from_file=temp_file_path)

  def test_zh_hant_normalize(self):
    indices = np.array([
      [6, 5, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
      [4, 8, 7, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ])
    strings = np.array([
      [u'您好嗎?'],
      [u'回頭見。']
    ])
    norm = n.StringTransform(
      col_index=0,
      language='zh_hant',
      max_vocab_size=20,
      max_sent_len=15,
    )
    norm.calc_global_values(strings)
    for i in xrange(2):
      array_dict = norm.forward_transform(strings)
      th.assert_arrays_equal(self, array_dict['data'], indices, threshold=0.1)

      out_array = norm.backward_transform(array_dict)
      th.assert_arrays_equal(self, out_array, strings)

      temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
      norm.save_to_file(temp_file_path)
      norm = n.StringTransform(from_file=temp_file_path)

  def test_zh_hant_normalize_2(self):
    indices = np.array([
      [3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
      [4, 6, 5, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ])
    strings = np.array([
      [u'您好嗎?'],
      [u'回頭見。']
    ])
    norm = n.StringTransform(
      col_index=0,
      language='zh_hant',
      max_vocab_size=20,
      max_sent_len=15,
      normalizer_kwargs={'half_width': True, 'remove_stopwords': True}
    )
    norm.calc_global_values(strings)
    for i in xrange(2):
      array_dict = norm.forward_transform(strings)
      th.assert_arrays_equal(self, array_dict['data'], indices, threshold=0.1)

      out_array = norm.backward_transform(array_dict)
      th.assert_arrays_equal(self, out_array, strings)

      temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
      norm.save_to_file(temp_file_path)
      norm = n.StringTransform(from_file=temp_file_path)

  def test_ko_normalize(self):
    indices = np.array([
      [7, 5, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
      [9, 1, 2, 8, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ])
    strings = np.array([
      [u'오류보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!'],
      [u'질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요.']
    ])
    norm = n.StringTransform(
      col_index=0,
      language='ko',
      max_vocab_size=20,
      max_sent_len=15,
    )
    norm.calc_global_values(strings)
    for i in xrange(2):
      array_dict = norm.forward_transform(strings)
      th.assert_arrays_equal(self, array_dict['data'], indices, threshold=0.1)

      out_array = norm.backward_transform(array_dict)
      th.assert_arrays_equal(self, out_array, strings)

      temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
      norm.save_to_file(temp_file_path)
      norm = n.StringTransform(from_file=temp_file_path)

  def test_ko_normalize_2(self):
    indices = np.array([
      [7, 5, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
      [9, 1, 2, 8, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ])
    strings = np.array([
      [u'오류보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!'],
      [u'질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요.']
    ])
    norm = n.StringTransform(
      col_index=0,
      language='ko',
      max_vocab_size=20,
      max_sent_len=15,
      normalizer_kwargs={'half_width': True, 'remove_stopwords': True}
    )
    norm.calc_global_values(strings)
    for i in xrange(2):
      array_dict = norm.forward_transform(strings)
      th.assert_arrays_equal(self, array_dict['data'], indices, threshold=0.1)

      out_array = norm.backward_transform(array_dict)
      th.assert_arrays_equal(self, out_array, strings)

      temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
      norm.save_to_file(temp_file_path)
      norm = n.StringTransform(from_file=temp_file_path)
if __name__ == "__main__":
  unittest.main()
