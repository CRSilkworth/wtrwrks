# -*- coding: utf-8 -*-
import shutil
import tempfile
import unittest
import production.utils.test_helpers as th
import production.transforms as n
import reversible_transforms.transforms.transform_set as ns
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
      [5, 11, 18, 12, 11, 3, -1, -1, -1, -1],
      [8, 2, 4, 1, 13, 10, 15, 17, 19, 3],
      [7, 16, 11, 14, 20, 2, 12, 1, 9, 3]
    ])
    new_strings = [
      "It is what it is .",
      "Whatever , Bob 's mother has seen the world .",
      "The sun is not yellow , it 's chicken ."
    ]
    strings = [
      "It is what it is.",
      "Whatever, Bob's mother has seen the world.",
      "The sun is not yellow, it's chicken. OK."
    ]
    df = pd.DataFrame({'a': strings})
    norm = n.StringTransform(
      df=df,
      columns=['a'],
      language='en'
    )
    for num, (index, row) in enumerate(df.iterrows()):
      vector = norm.row_to_vector(row)
      th.assert_arrays_equal(self, vector, indices[num], threshold=0.1)

      row = norm.vector_to_row(vector)
      self.assertEqual(row['a'], new_strings[num])

  def test_en_normalize_2(self):
    indices = np.array([
      [3, -1, -1, -1, -1, -1, -1, -1, -1, -1],
      [10, 2, 4, 1, 6, 8, 11, 3, -1, -1],
      [9, 12, 2, 1, 5, 3, 7, 3, -1, -1]
    ])
    new_strings = [
      ".",
      "whatever , bob 's mother see world .",
      "sun yellow , 's chicken . ok ."
    ]
    strings = [
      "It is what it is.",
      "Whatever, Bob's mother has seen the world.",
      "The sun is not yellow, it's chicken. OK."
    ]
    df = pd.DataFrame({'a': strings})
    norm = n.StringTransform(
      df=df,
      columns=['a'],
      language='en',
      lower=True,
      lemmatize=True,
      remove_stopwords=True
    )
    for num, (index, row) in enumerate(df.iterrows()):
      vector = norm.row_to_vector(row)
      th.assert_arrays_equal(self, vector, indices[num], threshold=0.1)

      row = norm.vector_to_row(vector)
      self.assertEqual(row['a'], new_strings[num])

  def test_ja_normalize(self):
    indices = np.array([
      [13, 14, 17, 18, 23, 8, 22, 20, 9, 15, 16, 3, 21, -1, -1],
      [6, 12, 4, 1, 11, 15, 19, 7, 5, 10, 2, -1, -1, -1, -1]
    ])
    new_strings = [
      u'チラシ ・ 勧誘 印刷 物 の 無断 投函 は 一 切 お 断り',
      u'すみませ ん が 、 もう 一 度 どお願い し ます 。'
    ]
    strings = [
      u'チラシ・勧誘印刷物の無断投函は一切お断り',
      u'すみませんが、もう一度どお願いします。'
    ]
    df = pd.DataFrame({'a': strings})
    norm = n.StringTransform(
      df=df,
      columns=['a'],
      language='ja',
      vector_size=15
    )
    for num, (index, row) in enumerate(df.iterrows()):
      vector = norm.row_to_vector(row)
      th.assert_arrays_equal(self, vector, indices[num], threshold=0.1)

      row = norm.vector_to_row(vector)
      self.assertEqual(row['a'], new_strings[num])

  def test_ja_normalize_2(self):
    indices = np.array([
      [11, 12, 15, 16, 21, 6, 20, 18, 7, 13, 14, 1, 19, -1, -1],
      [4, 10, 2, 9, 13, 17, 5, 3, 8, -1, -1, -1, -1, -1, -1]
    ])
    new_strings = [
      u'チラシ ・ 勧誘 印刷 物 の 無断 投函 は 一 切 お 断り',
      u'すみませ ん が もう 一 度 どお願い し ます'
    ]
    strings = [
      u'チラシ・勧誘印刷物の無断投函は一切お断り',
      u'すみませんが、もう一度どお願いします。'
    ]
    df = pd.DataFrame({'a': strings})
    norm = n.StringTransform(
      df=df,
      columns=['a'],
      language='ja',
      vector_size=15,
      half_width=True,
      remove_stopwords=True
    )
    for num, (index, row) in enumerate(df.iterrows()):
      vector = norm.row_to_vector(row)
      th.assert_arrays_equal(self, vector, indices[num], threshold=0.1)
      row = norm.vector_to_row(vector)
      self.assertEqual(row['a'], new_strings[num])

  def test_zh_hans_normalize(self):
    indices = np.array([
      [11, 4, 1, 5, 8, 6, 3, -1, -1, -1, -1, -1, -1, -1, -1],
      [10, 9, 1, 5, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ])
    new_strings = [
      u'早 上好 , 你 好 吗 。',
      u'我 很好 , 你 呢 ?'
    ]
    strings = [
      u'早上好,你好吗。',
      u'我很好,你呢?'
    ]
    df = pd.DataFrame({'a': strings})
    norm = n.StringTransform(
      df=df,
      columns=['a'],
      language='zh_hans',
      vector_size=15,
    )
    for num, (index, row) in enumerate(df.iterrows()):
      vector = norm.row_to_vector(row)
      th.assert_arrays_equal(self, vector, indices[num], threshold=0.1)
      row = norm.vector_to_row(vector)
      self.assertEqual(row['a'], new_strings[num])

  def test_zh_hans_normalize_2(self):
    indices = np.array([
      [5, 3, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
      [4, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ])
    new_strings = [
      u'早 上好 ,',
      u'很好 , ?'
    ]
    strings = [
      u'早上好,你好吗。',
      u'我很好,你呢?'
    ]
    df = pd.DataFrame({'a': strings})
    norm = n.StringTransform(
      df=df,
      columns=['a'],
      language='zh_hans',
      vector_size=15,
      half_width=True,
      remove_stopwords=True
    )
    for num, (index, row) in enumerate(df.iterrows()):
      vector = norm.row_to_vector(row)
      th.assert_arrays_equal(self, vector, indices[num], threshold=0.1)
      row = norm.vector_to_row(vector)

      self.assertEqual(row['a'], new_strings[num])

  def test_zh_hant_normalize(self):
    indices = np.array([
      [6, 5, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
      [4, 8, 7, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ])
    new_strings = [
      u'您 好 嗎 ?',
      u'回 頭 見 '
    ]
    strings = [
      u'您好嗎?',
      u'回頭見。'
    ]
    df = pd.DataFrame({'a': strings})
    norm = n.StringTransform(
      df=df,
      columns=['a'],
      language='zh_hant',
      vector_size=15,
    )
    for num, (index, row) in enumerate(df.iterrows()):
      vector = norm.row_to_vector(row)
      th.assert_arrays_equal(self, vector, indices[num], threshold=0.1)
      row = norm.vector_to_row(vector)
      self.assertEqual(row['a'], new_strings[num])

  def test_zh_hant_normalize_2(self):
    indices = np.array([
      [3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
      [4, 6, 5, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ])
    new_strings = [
      u'嗎 ?',
      u'回 頭 見 '
    ]
    strings = [
      u'您好嗎?',
      u'回頭見。'
    ]
    df = pd.DataFrame({'a': strings})
    norm = n.StringTransform(
      df=df,
      columns=['a'],
      language='zh_hant',
      vector_size=15,
      half_width=True,
      remove_stopwords=True
    )
    for num, (index, row) in enumerate(df.iterrows()):
      vector = norm.row_to_vector(row)
      th.assert_arrays_equal(self, vector, indices[num], threshold=0.1)
      row = norm.vector_to_row(vector)
      self.assertEqual(row['a'], new_strings[num])

  def test_ko_normalize(self):
    indices = np.array([
      [18, 9, 7, 14, 29, 2, 17, 8, 19, 28, 12, 21, 26, 11, 1],
      [25, 22, 4, 10, 20, 5, 23, 27, 16, 6, 15, 24, 13, 3, -1]
    ])
    new_strings = [
      u'오류 보고 는 실행 환경 , 에러 메세지 와 함께 설명 을 최대한 상세히 !',
      u'질문 이나 건의 사항 은 깃헙 이슈 트래커 에 남기 어 주 세요 .'
    ]
    strings = [
      u'오류보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!',
      u'질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요.'
    ]
    df = pd.DataFrame({'a': strings})
    norm = n.StringTransform(
      df=df,
      columns=['a'],
      language='ko',
      vector_size=15,
    )
    for num, (index, row) in enumerate(df.iterrows()):
      vector = norm.row_to_vector(row)
      th.assert_arrays_equal(self, vector, indices[num], threshold=0.1)
      row = norm.vector_to_row(vector)
      self.assertEqual(row['a'], new_strings[num])

  def test_ko_normalize_2(self):
    indices = np.array([
      [13, 6, 4, 11, 21, 12, 5, 9, 19, 8, -1, -1, -1, -1, -1],
      [18, 15, 1, 7, 14, 2, 16, 20, 3, 17, 10, -1, -1, -1, -1]
    ])
    new_strings = [
      u'오류 보고 는 실행 환경 에러 메세지 설명 최대한 상세히',
      u'질문 이나 건의 사항 은 깃헙 이슈 트래커 남기 주 세요'
    ]
    strings = [
      u'오류보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!',
      u'질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요.'
    ]
    df = pd.DataFrame({'a': strings})
    norm = n.StringTransform(
      df=df,
      columns=['a'],
      language='ko',
      vector_size=15,
      half_width=True,
      remove_stopwords=True
    )
    for num, (index, row) in enumerate(df.iterrows()):
      vector = norm.row_to_vector(row)
      th.assert_arrays_equal(self, vector, indices[num], threshold=0.1)
      row = norm.vector_to_row(vector)
      self.assertEqual(row['a'], new_strings[num])
if __name__ == "__main__":
  unittest.main()
