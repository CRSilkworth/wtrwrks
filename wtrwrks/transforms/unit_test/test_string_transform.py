# -*- coding: utf-8 -*-
import shutil
import tempfile
import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.transforms.string_transform as n
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
    indices = np.array([[[5, 11, 18, 12, 11, 3, -1, -1, -1, -1]], [[8, 2, 4, 1, 13, 10, 15, 17, 19, 3]], [[7, 16, 11, 14, 20, 2, 12, 1, 9, 3]]])

    strings = np.array([
      ["It is what it is."],
      ["Whatever, Bob's mother has seen the world."],
      ["The sun is not yellow, it's chicken. OK."]
    ])
    tokenize_diff = [['[["d", 16, 17, ""], ["d", 18, 22, ""]]'], ['[["d", 8, 9, ""], ["d", 14, 15, ""], ["d", 43, 44, ""]]'], ['[["d", 21, 22, ""], ["d", 26, 27, ""], ["i", 37, 37, "."], ["i", 38, 38, "OK"]]']]
    missing_vals = np.array([[['', '', '', '', '', '', '', '', '', '']], [['', '', '', '', '', '', '', '', '', '']], [['', '', '', '', '', '', '', '', '', '']]], dtype='|S42')
    index_to_word = ['[UNK]'] + self._get_index_to_word(strings, en_tokenizer)
    trans = n.StringTransform(
      index_to_word=index_to_word,
      word_tokenizer=en_tokenizer,
      name='string_transform',
      max_sent_len=10,
    )
    trans.calc_global_values(strings)
    for i in xrange(2):
      self.pour_pump(
        trans,
        strings,
        {
          'string_transform/indices': indices,
          'string_transform/missing_vals': missing_vals,
          'string_transform/tokenize_diff': tokenize_diff,

        },
        test_type=False
      )
      trans = self.write_read(trans, self.temp_dir)

  def test_array_iter(self):
    indices = np.array([[[0, 10, 17, 11, 10, 3, -1, -1, -1, -1]], [[7, 2, 4, 1, 12, 9, 14, 16, 18, 3]], [[6, 15, 10, 13, 19, 2, 11, 1, 8, 3]]])

    strings = np.array([
      ["It is what it is."],
      ["Whatever, Bob's mother has seen the world."],
      ["The sun is not yellow, it's chicken. OK."]
    ])
    tokenize_diff = [['[["d", 16, 17, ""], ["d", 18, 22, ""]]'], ['[["d", 8, 9, ""], ["d", 14, 15, ""], ["d", 43, 44, ""]]'], ['[["d", 21, 22, ""], ["d", 26, 27, ""], ["i", 37, 37, "."], ["i", 38, 38, "OK"]]']]
    missing_vals = np.array([[['It', '', '', '', '', '', '', '', '', '']], [['', '', '', '', '', '', '', '', '', '']], [['', '', '', '', '', '', '', '', '', '']]], dtype='|S42')
    # index_to_word = ['[UNK]'] + self._get_index_to_word(strings, en_tokenizer)
    trans = n.StringTransform(
      # index_to_word=index_to_word,
      word_tokenizer=en_tokenizer,
      name='string_transform',
      max_sent_len=10,
      max_vocab_size=20
    )

    array_iter = [strings[0: 1], strings[1: 2], strings[2: 3]]

    trans.calc_global_values(data_iter=array_iter)
    for i in xrange(2):
      self.pour_pump(
        trans,
        strings,
        {
          'string_transform/indices': indices,
          'string_transform/missing_vals': missing_vals,
          'string_transform/tokenize_diff': tokenize_diff,

        },
        test_type=False
      )
      trans = self.write_read(trans, self.temp_dir)

  def test_ja_normalize(self):

    indices = np.array([[[13, 14, 17, 18, 23, 8, 22, 20, 9, 15, 16, 3, 21, -1, -1]], [[6, 12, 4, 1, 11, 15, 19, 7, 5, 10, 2, -1, -1, -1, -1]]])
    strings = np.array([
      [u'チラシ・勧誘印刷物の無断投函は一切お断り'],
      [u'すみませんが、もう一度どお願いします。']
    ])
    tokenize_diff = [['[]'], ['[]']]
    missing_vals = np.array([[[u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'']], [[u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'']]], dtype='|U20')
    index_to_word = ['__UNK__'] + self._get_index_to_word(strings, ja_tokenizer)
    trans = n.StringTransform(
      name='',
      word_tokenizer=ja_tokenizer,
      index_to_word=index_to_word,
      word_detokenizer=lambda a: ''.join(a),
      max_sent_len=15,
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

        },
        test_type=False
      )
      trans = self.write_read(trans, self.temp_dir)

  def test_ja_normalize_2(self):

    indices = np.array([[[2, 1, 15, 16, 19, 20, 25, 10, 24, 22, 11, 17, 18, 5, 23]], [[8, 14, 6, 3, 13, 17, 21, 9, 7, 12, 4, -1, -1, -1, -1]]])
    strings = np.array([
      [u'２０チラシ・勧誘印刷物の無断投函は一切お断り'],
      [u'すみませんが、もう一度どお願いします。']
    ])
    tokenize_diff = [['[]'], ['[]']]
    missing_vals = np.array([[[u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'']], [[u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'', u'']]], dtype='|U20')
    index_to_word = ['[UNK]'] + self._get_index_to_word(strings, ja_tokenizer, half_width=True)
    half_width_diff = [[['[["i", 0, 1, "\\uff12"]]', '[["i", 0, 1, "\\uff10"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']], [['[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']]]
    trans = n.StringTransform(
      name='',
      word_tokenizer=ja_tokenizer,
      index_to_word=index_to_word,
      word_detokenizer=lambda a: ''.join(a),
      half_width=True,
      max_sent_len=15,
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

        },
        test_type=False
      )
      trans = self.write_read(trans, self.temp_dir)

  def test_read_write(self):
    indices = np.array([
      [
        [16, 15, 27, 16, 15, 5, -1, -1, -1, -1],
        [13, 3, 20, 22, 7, 13, 3, 20, 19, -1]
      ],
      [
        [24, 23, 15, 17, 28, 4, 16, 2, 9, 5],
        [12, 4, 0, 1, -1, -1, -1, -1, -1, -1]
      ],
      [
        [11, 26, 21, 14, 6, 10, 5, -1, -1, -1],
        [25, 6, 8, -1, -1, -1, -1, -1, -1, -1]
      ]
    ])
    lower_case_diff = [
      [
        ['[["i", 0, 1, "I"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'],
        ['[["i", 0, 1, "I"]]', '[]', '[]', '[]', '[]', '[["i", 0, 1, "I"]]', '[]', '[]', '[]', '[]']],
      [
        ['[["i", 0, 1, "T"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'],
        ['[["i", 0, 1, "H"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']],
      [
        ['[["i", 0, 1, "E"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'],
        ['[["i", 0, 1, "U"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']
      ]
    ]
    tokenize_diff = [
      [
        '[["d", 16, 17, ""], ["d", 18, 22, ""]]',
        '[["d", 1, 2, ""], ["d", 23, 24, ""], ["d", 37, 38, ""]]'
      ],
      [
        '[["d", 21, 22, ""], ["d", 26, 27, ""], ["i", 37, 37, "."], ["i", 38, 38, "OK"]]',
        '[["d", 3, 4, ""], ["d", 9, 10, ""], ["d", 11, 17, ""]]'
      ],
      [
        '[["d", 30, 31, ""], ["d", 32, 35, ""]]',
        '[["d", 14, 21, ""]]'
      ]
    ]
    missing_vals = np.array([[[u'', u'', u'', u'', u'', u'', u'', u'', u'', u''], [u'', u'', u'', u'', u'', u'', u'', u'', u'', u'']], [[u'', u'', u'', u'', u'', u'', u'', u'', u'', u''], [u'', u'', u'you', u'', u'', u'', u'', u'', u'', u'']], [[u'', u'', u'', u'', u'', u'', u'', u'', u'', u''], [u'', u'', u'', u'', u'', u'', u'', u'', u'', u'']]], dtype='|S40')
    strings = np.array([
      ["It is what it is.", "I've seen summer and I've seen rain"],
      ["The sun is not yellow, it's chicken. OK.", "Hey, you!"],
      ['Ended up sleeping in a doorway.', 'Under a bodega']
    ], dtype=np.unicode)
    index_to_word = self._get_index_to_word(np.char.lower(strings), en_tokenizer)

    index_to_word = ['[UNK]'] + index_to_word[:-1]

    trans = n.StringTransform(
      name='',
      index_to_word=index_to_word,
      word_tokenizer=en_tokenizer,
      lower_case=True,
      max_sent_len=10,
    )
    trans.calc_global_values(strings)
    # for i in xrange(2):
    self.pour_pump(
      trans,
      strings,
      {
        'indices': indices,
        'missing_vals': missing_vals,
        'tokenize_diff': tokenize_diff,
        'lower_case_diff': lower_case_diff

      },
      test_type=False
    )

    self.write_read_example(trans, strings, self.temp_dir)
    trans = self.write_read(trans, self.temp_dir)

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
