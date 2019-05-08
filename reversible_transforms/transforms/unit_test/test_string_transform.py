# -*- coding: utf-8 -*-
import shutil
import tempfile
import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.transforms as n
from chop.mmseg import Tokenizer as MMSEGTokenizer
from chop.hmm import Tokenizer as HMMTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import tinysegmenter
import os
import pandas as pd
import numpy as np
import unicodedata

en_tokenizer = nltk.word_tokenize
ko_tokenizer = lambda s: s.split()
zh_hant_tokenizer = MMSEGTokenizer().cut
ts = tinysegmenter.TinySegmenter()
ja_tokenizer = ts.tokenize
zh_hans_tokenizer = HMMTokenizer().cut

word_net_lemmatizer = WordNetLemmatizer()

def _half_width(string):
  return unicodedata.normalize('NFKC', unicode(string))

def get_wordnet_pos(treebank_tag):
  if treebank_tag.startswith('J'):
    return wordnet.ADJ
  elif treebank_tag.startswith('V'):
    return wordnet.VERB
  elif treebank_tag.startswith('N'):
    return wordnet.NOUN
  elif treebank_tag.startswith('R'):
    return wordnet.ADV
  else:
    return wordnet.NOUN


def basic_lemmatizer(string):
  if string == '':
    return string

  pos_tags = nltk.pos_tag([string])

  if not pos_tags:
    return string

  pos_tag = get_wordnet_pos(pos_tags[0][1])
  lemma = word_net_lemmatizer.lemmatize(string, pos_tag)

  return lemma


class TestStringTransform(th.TestTransform):
  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir)

  def test_en_normalize(self):
    indices = np.array([[[4, 10, 17, 11, 10, 2, -1, -1, -1, -1]], [[7, 1, 3, 0, 12, 9, 14, 16, 18, 2]], [[6, 15, 10, 13, 19, 1, 11, 0, 8, 2]]])

    strings = np.array([
      ["It is what it is."],
      ["Whatever, Bob's mother has seen the world."],
      ["The sun is not yellow, it's chicken. OK."]
    ])
    tokenize_diff = [['[["d", 16, 17, ""], ["d", 18, 22, ""]]'], ['[["d", 8, 9, ""], ["d", 14, 15, ""], ["d", 43, 44, ""]]'], ['[["d", 21, 22, ""], ["d", 26, 27, ""], ["i", 37, 37, "."], ["i", 38, 38, "OK"]]']]
    missing_vals = np.array(['', '', '', ''], dtype='|S8')
    index_to_word = self._get_index_to_word(strings, en_tokenizer)
    trans = n.StringTransform(
      index_to_word=index_to_word,
      tokenizer=en_tokenizer,
      name='string_transform',
      max_sent_len=10,
    )
    trans.calc_global_values(strings)
    for i in xrange(2):
      self.pour_pump(
        trans,
        strings,
        {
          'indices': indices,
          'missing_vals': missing_vals,
          'tokenize_diff': tokenize_diff,

        }
      )
      trans = self.write_read(trans, self.temp_dir)
  #
  def test_en_normalize_2(self):

    indices = np.array([
      [[8, 4, 15, 8, 4, 3, -1, -1, -1, -1]],
      [[16, 2, 5, 1, 9, 7, 12, 14, 17, 3]],
      [[14, 13, 4, 10, 18, 2, 8, 1, 6, 3]]
    ])
    lemmatize_diff = [[['[]', '[["i", 0, 2, "is"]]', '[]', '[]', '[["i", 0, 2, "is"]]', '[]', '[]', '[]', '[]', '[]']], [['[]', '[]', '[]', '[]', '[]', '[["i", 2, 4, "s"]]', '[["i", 3, 3, "n"]]', '[]', '[]', '[]']], [['[]', '[]', '[["i", 0, 2, "is"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']]]
    lower_case_diff = [[['[["i", 0, 1, "I"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']], [['[["i", 0, 1, "W"]]', '[]', '[["i", 0, 1, "B"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']], [['[["i", 0, 1, "T"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']]]
    tokenize_diff = [['[["d", 16, 17, ""], ["d", 18, 22, ""]]'], ['[["d", 8, 9, ""], ["d", 14, 15, ""], ["d", 43, 44, ""]]'], ['[["d", 21, 22, ""], ["d", 26, 27, ""], ["i", 37, 37, "."], ["i", 38, 38, "OK"]]']]
    missing_vals = np.array([], dtype='|S8')
    strings = np.array([
      ["It is what it is."],
      ["Whatever, Bob's mother has seen the world."],
      ["The sun is not yellow, it's chicken. OK."]
    ])
    index_to_word = self._get_index_to_word(np.char.lower(strings), en_tokenizer, basic_lemmatizer)
    index_to_word = ['__UNK__'] + index_to_word
    trans = n.StringTransform(
      index_to_word=index_to_word,
      tokenizer=en_tokenizer,
      lemmatize=True,
      lemmatizer=basic_lemmatizer,
      lower_case=True,
      unk_index=0,
      max_sent_len=10,
    )
    trans.calc_global_values(strings)
    for i in xrange(2):
      self.pour_pump(
        trans,
        strings,
        {
          'indices': indices,
          'lower_case_diff': lower_case_diff,
          'lemmatize_diff': lemmatize_diff,
          'missing_vals': missing_vals,
          'tokenize_diff': tokenize_diff,

        }
      )
      trans = self.write_read(trans, self.temp_dir)

  def test_ja_normalize(self):

    indices = np.array([[[12, 13, 16, 17, 22, 7, 21, 19, 8, 14, 15, 2, 20, -1, -1]], [[5, 11, 3, 0, 10, 14, 18, 6, 4, 9, 1, -1, -1, -1, -1]]])
    strings = np.array([
      [u'チラシ・勧誘印刷物の無断投函は一切お断り'],
      [u'すみませんが、もう一度どお願いします。']
    ])
    tokenize_diff = [['[]'], ['[]']]
    missing_vals = np.array(['', '', '', '', '', ''], dtype='|S8')
    index_to_word = self._get_index_to_word(strings, ja_tokenizer)
    trans = n.StringTransform(
      tokenizer=ja_tokenizer,
      index_to_word=index_to_word,
      delimiter='',
      max_sent_len=15
    )
    trans.calc_global_values(strings)
    for i in xrange(2):
      self.pour_pump(
        trans,
        strings,
        {
          'indices': indices,
          'missing_vals': missing_vals,
          'tokenize_diff': tokenize_diff,

        }
      )
      trans = self.write_read(trans, self.temp_dir)

  def test_ja_normalize_2(self):

    indices = np.array([[[1, 0, 14, 15, 18, 19, 24, 9, 23, 21, 10, 16, 17, 4, 22]], [[7, 13, 5, 2, 12, 16, 20, 8, 6, 11, 3, -1, -1, -1, -1]]])
    strings = np.array([
      [u'２０チラシ・勧誘印刷物の無断投函は一切お断り'],
      [u'すみませんが、もう一度どお願いします。']
    ])
    tokenize_diff = [['[]'], ['[]']]
    missing_vals = np.array(['', '', '', ''], dtype='|S8')
    index_to_word = self._get_index_to_word(strings, ja_tokenizer, half_width=True)
    half_width_diff = [[['[["i", 0, 1, "\\uff12"]]', '[["i", 0, 1, "\\uff10"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']], [['[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']]]
    trans = n.StringTransform(
      tokenizer=ja_tokenizer,
      index_to_word=index_to_word,
      delimiter='',
      half_width=True,
      max_sent_len=15
    )
    trans.calc_global_values(strings)
    for i in xrange(2):
      self.pour_pump(
        trans,
        strings,
        {
          'indices': indices,
          'missing_vals': missing_vals,
          'tokenize_diff': tokenize_diff,
          'half_width_diff': half_width_diff

        }
      )
      trans = self.write_read(trans, self.temp_dir)


  # def test_ja_normalize_2(self):
  #   indices = np.array([
  #     [10, 11, 0, 14, 19, 6, 18, 16, 7, 12, 13, 1, 17, -1, -1],
  #     [4, 9, 2, 0, 12, 15, 5, 3, 8, -1, -1, -1, -1, -1, -1]
  #   ])
  #
  #   strings = np.array([
  #     [u'２０チラシ・勧誘印刷物の無断投函は一切お断り'],
  #     [u'すみませんが、もう一度どお願いします。']
  #   ])
  #   trans = n.StringTransform(
  #     col_index=0,
  #     language='ja',
  #     max_vocab_size=20,
  #     max_sent_len=15,
  #     normalizer_kwargs={'half_width': True, 'remove_stopwords': True}
  #   )
  #   trans.calc_global_values(strings)
  #   for i in xrange(2):
  #     array_dict = trans.forward_transform(strings)
  #     th.assert_arrays_equal(self, array_dict['data'], indices, threshold=0.1)
  #
  #     out_array = trans.backward_transform(array_dict)
  #     th.assert_arrays_equal(self, out_array, strings)
  #
  #     temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
  #     trans.save_to_file(temp_file_path)
  #     trans = n.StringTransform(from_file=temp_file_path)
  #
  # def test_zh_hans_normalize(self):
  #   indices = np.array([
  #     [11, 4, 1, 5, 8, 6, 3, -1, -1, -1, -1, -1, -1, -1, -1],
  #     [10, 9, 1, 5, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  #   ])
  #   strings = np.array([
  #     [u'早上好,你好吗。'],
  #     [u'我很好,你呢?']
  #   ])
  #   trans = n.StringTransform(
  #     col_index=0,
  #     language='zh_hans',
  #     max_vocab_size=20,
  #     max_sent_len=15,
  #   )
  #   trans.calc_global_values(strings)
  #   for i in xrange(2):
  #     array_dict = trans.forward_transform(strings)
  #     th.assert_arrays_equal(self, array_dict['data'], indices, threshold=0.1)
  #
  #     out_array = trans.backward_transform(array_dict)
  #     th.assert_arrays_equal(self, out_array, strings)
  #
  #     temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
  #     trans.save_to_file(temp_file_path)
  #     trans = n.StringTransform(from_file=temp_file_path)
  #
  # def test_zh_hans_normalize_2(self):
  #   indices = np.array([
  #     [5, 3, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  #     [4, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  #   ])
  #   strings = np.array([
  #     [u'早上好,你好吗。'],
  #     [u'我很好,你呢?']
  #   ])
  #   trans = n.StringTransform(
  #     col_index=0,
  #     language='zh_hans',
  #     max_vocab_size=20,
  #     max_sent_len=15,
  #     normalizer_kwargs={'half_width': True, 'remove_stopwords': True}
  #   )
  #   trans.calc_global_values(strings)
  #   for i in xrange(2):
  #     array_dict = trans.forward_transform(strings)
  #     th.assert_arrays_equal(self, array_dict['data'], indices, threshold=0.1)
  #
  #     out_array = trans.backward_transform(array_dict)
  #     th.assert_arrays_equal(self, out_array, strings)
  #
  #     temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
  #     trans.save_to_file(temp_file_path)
  #     trans = n.StringTransform(from_file=temp_file_path)
  #
  # def test_zh_hant_normalize(self):
  #   indices = np.array([
  #     [6, 5, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  #     [4, 8, 7, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  #   ])
  #   strings = np.array([
  #     [u'您好嗎?'],
  #     [u'回頭見。']
  #   ])
  #   trans = n.StringTransform(
  #     col_index=0,
  #     language='zh_hant',
  #     max_vocab_size=20,
  #     max_sent_len=15,
  #   )
  #   trans.calc_global_values(strings)
  #   for i in xrange(2):
  #     array_dict = trans.forward_transform(strings)
  #     th.assert_arrays_equal(self, array_dict['data'], indices, threshold=0.1)
  #
  #     out_array = trans.backward_transform(array_dict)
  #     th.assert_arrays_equal(self, out_array, strings)
  #
  #     temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
  #     trans.save_to_file(temp_file_path)
  #     trans = n.StringTransform(from_file=temp_file_path)
  #
  # def test_zh_hant_normalize_2(self):
  #   indices = np.array([
  #     [3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  #     [4, 6, 5, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  #   ])
  #   strings = np.array([
  #     [u'您好嗎?'],
  #     [u'回頭見。']
  #   ])
  #   trans = n.StringTransform(
  #     col_index=0,
  #     language='zh_hant',
  #     max_vocab_size=20,
  #     max_sent_len=15,
  #     normalizer_kwargs={'half_width': True, 'remove_stopwords': True}
  #   )
  #   trans.calc_global_values(strings)
  #   for i in xrange(2):
  #     array_dict = trans.forward_transform(strings)
  #     th.assert_arrays_equal(self, array_dict['data'], indices, threshold=0.1)
  #
  #     out_array = trans.backward_transform(array_dict)
  #     th.assert_arrays_equal(self, out_array, strings)
  #
  #     temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
  #     trans.save_to_file(temp_file_path)
  #     trans = n.StringTransform(from_file=temp_file_path)
  #
  # def test_ko_normalize(self):
  #   indices = np.array([
  #     [7, 5, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  #     [9, 1, 2, 8, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  #   ])
  #   strings = np.array([
  #     [u'오류보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!'],
  #     [u'질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요.']
  #   ])
  #   trans = n.StringTransform(
  #     col_index=0,
  #     language='ko',
  #     max_vocab_size=20,
  #     max_sent_len=15,
  #   )
  #   trans.calc_global_values(strings)
  #   for i in xrange(2):
  #     array_dict = trans.forward_transform(strings)
  #     th.assert_arrays_equal(self, array_dict['data'], indices, threshold=0.1)
  #
  #     out_array = trans.backward_transform(array_dict)
  #     th.assert_arrays_equal(self, out_array, strings)
  #
  #     temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
  #     trans.save_to_file(temp_file_path)
  #     trans = n.StringTransform(from_file=temp_file_path)
  #
  # def test_ko_normalize_2(self):
  #   indices = np.array([
  #     [7, 5, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  #     [9, 1, 2, 8, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  #   ])
  #   strings = np.array([
  #     [u'오류보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!'],
  #     [u'질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요.']
  #   ])
  #   trans = n.StringTransform(
  #     col_index=0,
  #     language='ko',
  #     max_vocab_size=20,
  #     max_sent_len=15,
  #     normalizer_kwargs={'half_width': True, 'remove_stopwords': True}
  #   )
  #   trans.calc_global_values(strings)
  #   for i in xrange(2):
  #     array_dict = trans.forward_transform(strings)
  #     th.assert_arrays_equal(self, array_dict['data'], indices, threshold=0.1)
  #
  #     out_array = trans.backward_transform(array_dict)
  #     th.assert_arrays_equal(self, out_array, strings)
  #
  #     temp_file_path = os.path.join(self.temp_dir, 'temp.pickle')
  #     trans.save_to_file(temp_file_path)
  #     trans = n.StringTransform(from_file=temp_file_path)
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
