# -*- coding: utf-8 -*-
import shutil
import tempfile
import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.transforms.multi_lingual_string_transform as n
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

en_detokenizer = lambda s: ' '.join(s)
ja_detokenizer = lambda s: ''.join(s)

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


class TestMultiStringTransform(th.TestTransform):
  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir)

  def test_multi(self):
    indices = np.array([[[6, 12, 18, 13, 12, 4, -1, -1, -1, -1]], [[0, 3, 0, 1, 0, 0, 15, 0, 0, 4]], [[8, 17, 12, 14, 19, 3, 13, 1, 11, 4]], [[10, 0, 13, 14, 19, 0, 18, 16, 0, 11]], [[0, 9, 4, 1, 8, 11, 15, 6, 5, 7]], [[6, 12, 18, 13, 12, 4, -1, -1, -1, -1]], [[5, 2, 15, 16, 10, 5, 2, 15, 0, -1]], [[8, 17, 12, 14, 19, 3, 13, 1, 11, 4]], [[0, 3, 0, 0, -1, -1, -1, -1, -1, -1]], [[0, 0, 0, 0, 9, 0, 4, -1, -1, -1]], [[0, 9, 0, -1, -1, -1, -1, -1, -1, -1]], [[0, 0, 10, 0, 13, 14, 19, 0, 18, 16]], [[0, 9, 4, 1, 8, 11, 15, 6, 5, 7]]])
    missing_vals = np.array([[[u'', u'', u'', u'', u'', u'', u'', u'', u'', u'']], [[u'Whatever', u'', u'Bob', u'', u'mother', u'has', u'', u'the', u'world', u'']], [[u'', u'', u'', u'', u'', u'', u'', u'', u'', u'']], [[u'', u'\u30fb', u'', u'', u'', u'\u306e', u'', u'', u'\u306f', u'']], [[u'\u3059\u307f\u307e\u305b', u'', u'', u'', u'', u'', u'', u'', u'', u'']], [[u'', u'', u'', u'', u'', u'', u'', u'', u'', u'']], [[u'', u'', u'', u'', u'', u'', u'', u'', u'rain', u'']], [[u'', u'', u'', u'', u'', u'', u'', u'', u'', u'']], [[u'Hey', u'', u'you', u'!', u'', u'', u'', u'', u'', u'']], [[u'Ended', u'up', u'sleeping', u'in', u'', u'doorway', u'', u'', u'', u'']], [[u'Under', u'', u'bodega', u'', u'', u'', u'', u'', u'', u'']], [[u'\uff12', u'\uff10', u'', u'\u30fb', u'', u'', u'', u'\u306e', u'', u'']], [[u'\u3059\u307f\u307e\u305b', u'', u'', u'', u'', u'', u'', u'', u'', u'']]])
    tokenize_diff = np.array([[u'[["d", 16, 17, ""], ["d", 18, 22, ""]]'], [u'[["d", 8, 9, ""], ["d", 14, 15, ""], ["d", 43, 44, ""]]'], [u'[["d", 21, 22, ""], ["d", 26, 27, ""], ["i", 37, 37, "."], ["i", 38, 38, "OK"]]'], [u'[["i", 16, 16, "\\u5207\\u304a\\u65ad\\u308a"]]'], [u'[["i", 18, 18, "\\u3002"]]'], [u'[["d", 16, 17, ""], ["d", 18, 22, ""]]'], [u'[["d", 1, 2, ""], ["d", 23, 24, ""], ["d", 37, 38, ""]]'], [u'[["d", 21, 22, ""], ["d", 26, 27, ""], ["i", 37, 37, "."], ["i", 38, 38, "OK"]]'], [u'[["d", 3, 4, ""], ["d", 9, 10, ""], ["d", 11, 17, ""]]'], [u'[["d", 30, 31, ""], ["d", 32, 35, ""]]'], [u'[["d", 14, 21, ""]]'], [u'[["i", 16, 16, "\\u306f\\u4e00\\u5207\\u304a\\u65ad\\u308a"]]'], [u'[["i", 18, 18, "\\u3002"]]']])

    strings = np.array([
      [u"It is what it is."],
      [u"Whatever, Bob's mother has seen the world."],
      [u"The sun is not yellow, it's chicken. OK."],
      [u'チラシ・勧誘印刷物の無断投函は一切お断り'],
      [u'すみませんが、もう一度どお願いします。'],
      [u"It is what it is."],
      [u"I've seen summer and I've seen rain"],
      [u"The sun is not yellow, it's chicken. OK."],
      [u"Hey, you!"],
      [u'Ended up sleeping in a doorway.'],
      [u'Under a bodega'],
      [u'２０チラシ・勧誘印刷物の無断投函は一切お断り'],
      [u'すみませんが、もう一度どお願いします。']
    ])
    languages = np.array([['en'], ['en'], ['en'], ['ja'], ['ja'], ['en'], ['en'], ['en'], ['en'], ['en'], ['en'], ['ja'], ['ja']])

    array = np.concatenate((strings, languages), axis=1)

    # index_to_word = self._get_index_to_word(strings, en_tokenizer) + ['__UNK__']
    trans = n.MultiLingualStringTransform(
      word_tokenizers={'en': en_tokenizer, 'ja': ja_tokenizer},
      word_detokenizers={'en': en_detokenizer, 'ja': ja_detokenizer},
      name='string_transform',
      max_sent_len=10,
      max_vocab_size=20,
    )
    trans.calc_global_values(array)

    for i in xrange(2):
      self.pour_pump(
        trans,
        array,
        {
          'string_transform/indices': indices,
          'string_transform/languages': languages,
          'string_transform/missing_vals': missing_vals,
          'string_transform/tokenize_diff': tokenize_diff,

        },
        test_type=False
      )
      trans = self.write_read(trans, self.temp_dir)

  # def test_ja_normalize(self):
  #
  #   indices = np.array([[[12, 13, 16, 17, 22, 7, 21, 19, 8, 14, 15, 2, 20, -1, -1]], [[5, 11, 3, 0, 10, 14, 18, 6, 4, 9, 1, -1, -1, -1, -1]]])
  #
  #   tokenize_diff = [['[]'], ['[]']]
  #   missing_vals = np.array([[[u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'']], [[u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'']]], dtype='|U20')
  #   index_to_word = self._get_index_to_word(strings, ja_tokenizer) + ['__UNK__']
  #   trans = n.StringTransform(
  #     word_tokenizer=ja_tokenizer,
  #     index_to_word=index_to_word,
  #     word_detokenizer=lambda a: ''.join(a),
  #     max_sent_len=15,
  #     unk_index=len(index_to_word) - 1
  #   )
  #   trans.calc_global_values(strings)
  #   for i in xrange(2):
  #     self.pour_pump(
  #       trans,
  #       strings,
  #       {
  #         'indices': indices,
  #         'missing_vals': missing_vals,
  #         'tokenize_diff': tokenize_diff,
  #
  #       },
  #       test_type=False
  #     )
  #     trans = self.write_read(trans, self.temp_dir)
  #
  # def test_ja_normalize_2(self):
  #
  #   indices = np.array([[[1, 0, 14, 15, 18, 19, 24, 9, 23, 21, 10, 16, 17, 4, 22]], [[7, 13, 5, 2, 12, 16, 20, 8, 6, 11, 3, -1, -1, -1, -1]]])
  #
  #   tokenize_diff = [['[]'], ['[]']]
  #   missing_vals = np.array([[[u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'']], [[u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'']]], dtype='|U20')
  #   index_to_word = self._get_index_to_word(strings, ja_tokenizer, half_width=True) + ['__UNK__']
  #   half_width_diff = [[['[["i", 0, 1, "\\uff12"]]', '[["i", 0, 1, "\\uff10"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']], [['[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']]]
  #   trans = n.StringTransform(
  #     word_tokenizer=ja_tokenizer,
  #     index_to_word=index_to_word,
  #     word_detokenizer=lambda a: ''.join(a),
  #     half_width=True,
  #     max_sent_len=15,
  #     unk_index=len(index_to_word) - 1
  #   )
  #   trans.calc_global_values(strings)
  #   for i in xrange(2):
  #     self.pour_pump(
  #       trans,
  #       strings,
  #       {
  #         'indices': indices,
  #         'missing_vals': missing_vals,
  #         'tokenize_diff': tokenize_diff,
  #         'half_width_diff': half_width_diff
  #
  #       },
  #       test_type=False
  #     )
  #     trans = self.write_read(trans, self.temp_dir)
  #
  # def test_read_write(self):
  #   indices = np.array([
  #     [
  #       [16, 15, 27, 16, 15, 5, -1, -1, -1, -1],
  #       [13, 3, 20, 22, 7, 13, 3, 20, 19, -1]
  #     ],
  #     [
  #       [24, 23, 15, 17, 28, 4, 16, 2, 9, 5],
  #       [12, 4, 0, 1, -1, -1, -1, -1, -1, -1]
  #     ],
  #     [
  #       [11, 26, 21, 14, 6, 10, 5, -1, -1, -1],
  #       [25, 6, 8, -1, -1, -1, -1, -1, -1, -1]
  #     ]
  #   ])
  #   lower_case_diff = [
  #     [
  #       ['[["i", 0, 1, "I"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'],
  #       ['[["i", 0, 1, "I"]]', '[]', '[]', '[]', '[]', '[["i", 0, 1, "I"]]', '[]', '[]', '[]', '[]']],
  #     [
  #       ['[["i", 0, 1, "T"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'],
  #       ['[["i", 0, 1, "H"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']],
  #     [
  #       ['[["i", 0, 1, "E"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'],
  #       ['[["i", 0, 1, "U"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']
  #     ]
  #   ]
  #   tokenize_diff = [
  #     [
  #       '[["d", 16, 17, ""], ["d", 18, 22, ""]]',
  #       '[["d", 1, 2, ""], ["d", 23, 24, ""], ["d", 37, 38, ""]]'
  #     ],
  #     [
  #       '[["d", 21, 22, ""], ["d", 26, 27, ""], ["i", 37, 37, "."], ["i", 38, 38, "OK"]]',
  #       '[["d", 3, 4, ""], ["d", 9, 10, ""], ["d", 11, 17, ""]]'
  #     ],
  #     [
  #       '[["d", 30, 31, ""], ["d", 32, 35, ""]]',
  #       '[["d", 14, 21, ""]]'
  #     ]
  #   ]
  #   missing_vals = np.array([[[u'', u'', u'', u'', u'', u'', u'', u'', u'', u''], [u'', u'', u'', u'', u'', u'', u'', u'', u'', u'']], [[u'', u'', u'', u'', u'', u'', u'', u'', u'', u''], [u'', u'', u'you', u'', u'', u'', u'', u'', u'', u'']], [[u'', u'', u'', u'', u'', u'', u'', u'', u'', u''], [u'', u'', u'', u'', u'', u'', u'', u'', u'', u'']]], dtype='|S40')
  #   strings = np.array([
  #     ["It is what it is.", "I've seen summer and I've seen rain"],
  #     ["The sun is not yellow, it's chicken. OK.", "Hey, you!"],
  #     ['Ended up sleeping in a doorway.', 'Under a bodega']
  #   ], dtype=np.unicode)
  #   index_to_word = self._get_index_to_word(np.char.lower(strings), en_tokenizer)
  #
  #   index_to_word = ['__UNK__'] + index_to_word[:-1]
  #
  #   trans = n.StringTransform(
  #     index_to_word=index_to_word,
  #     word_tokenizer=en_tokenizer,
  #     lower_case=True,
  #     unk_index=0,
  #     max_sent_len=10,
  #   )
  #   trans.calc_global_values(strings)
  #   # for i in xrange(2):
  #   self.pour_pump(
  #     trans,
  #     strings,
  #     {
  #       'indices': indices,
  #       'missing_vals': missing_vals,
  #       'tokenize_diff': tokenize_diff,
  #       'lower_case_diff': lower_case_diff
  #
  #     },
  #     test_type=False
  #   )
  #
  #   self.write_read_example(trans, strings, self.temp_dir)
  #   trans = self.write_read(trans, self.temp_dir)

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
