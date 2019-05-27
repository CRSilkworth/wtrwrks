# -*- coding: utf-8 -*-
import shutil
import tempfile
import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.transforms.document_transform as dct
import wtrwrks.transforms.string_transform as st
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

  def test_keep_dims(self):
    indices = np.array([
    [[[18, 17, 36, 18, 17, -1, -1, -1, -1, -1],
       [15, 3, 28, 31, 7, 15, 3, 28, 27, -1],
       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
      [[33, 32, 17, 23, 37, 4, 18, 2, 10, -1],
       [0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
       [14, 4, 0, 1, -1, -1, -1, -1, -1, -1]]],
    [[[13, 35, 29, 16, 6, 12, -1, -1, -1, -1],
       [34, 6, 8, -1, -1, -1, -1, -1, -1, -1],
       [21, 26, 9, -1, -1, -1, -1, -1, -1, -1]],
      [[22, 25, 20, -1, -1, -1, -1, -1, -1, -1],
       [19, 30, 0, 11, -1, -1, -1, -1, -1, -1],
       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]]
    ])
    lower_case_diff = [[[['[["i", 0, 1, "I"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'], ['[["i", 0, 1, "I"]]', '[]', '[]', '[]', '[]', '[["i", 0, 1, "I"]]', '[]', '[]', '[]', '[]'], ['[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']], [['[["i", 0, 1, "T"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'], ['[["i", 0, 2, "OK"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'], ['[["i", 0, 1, "H"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']]], [[['[["i", 0, 1, "E"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'], ['[["i", 0, 1, "U"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'], ['[["i", 0, 1, "L"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']], [['[["i", 0, 1, "L"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'], ['[["i", 0, 1, "I"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'], ['[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']]]]
    tokenize_diff = [[['[["d", 16, 21, ""]]', '[["i", 0, 0, " "], ["d", 1, 2, ""], ["d", 23, 24, ""], ["d", 37, 38, ""]]', '[["d", 0, 9, ""]]'], ['[["d", 21, 22, ""], ["d", 26, 27, ""], ["d", 37, 38, ""]]', '[["i", 0, 0, " "], ["d", 2, 11, ""]]', '[["i", 0, 0, " "], ["d", 3, 4, ""], ["d", 9, 10, ""], ["d", 11, 17, ""]]']], [['[["d", 30, 34, ""]]', '[["i", 0, 0, " "], ["d", 14, 21, ""]]', '[["i", 0, 0, " "], ["d", 20, 27, ""]]'], ['[["d", 12, 19, ""]]', '[["i", 0, 0, " "], ["d", 21, 27, ""]]', '[["d", 0, 9, ""]]']]]
    missing_vals = np.array(['ok', 'you', 'you'], dtype='|S40')
    strings = np.array([
      [
        "It is what it is. I've seen summer and I've seen rain",
        "The sun is not yellow, it's chicken. OK. Hey, you!"
      ],
      [
        'Ended up sleeping in a doorway. Under a bodega. Lights over broadway',
        'Look out kid. Its something you did.'
      ]
    ], dtype=np.unicode)
    index_to_word = self._get_index_to_word(np.char.lower(strings), en_tokenizer)

    index_to_word = ['__UNK__'] + index_to_word[:-1]

    string_trans = st.StringTransform(
      index_to_word=index_to_word,
      word_tokenizer=en_tokenizer,
      lower_case=True,
      unk_index=0,
      max_sent_len=10,
      name='ST'
    )
    trans = dct.DocumentTransform(
      sent_tokenizer=lambda s: s.split('.'),
      string_transform=string_trans,
    )
    # for i in xrange(2):
    self.pour_pump(
      trans,
      strings,
      {
        'ST/indices': indices,
        'ST/missing_vals': missing_vals,
        'ST/tokenize_diff': tokenize_diff,
        'ST/lower_case_diff': lower_case_diff

      },
      test_type=False
    )

    self.write_read_example(trans, strings, self.temp_dir)
    trans = self.write_read(trans, self.temp_dir)

  def test_remove_dims(self):
    indices = np.array([
      [18, 17, 36, 18, 17, -1, -1, -1, -1, -1],
      [15, 3, 28, 31, 7, 15, 3, 28, 27, -1],
      [33, 32, 17, 23, 37, 4, 18, 2, 10, -1],
      [0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
      [14, 4, 0, 1, -1, -1, -1, -1, -1, -1],
      [13, 35, 29, 16, 6, 12, -1, -1, -1, -1],
      [34, 6, 8, -1, -1, -1, -1, -1, -1, -1],
      [21, 26, 9, -1, -1, -1, -1, -1, -1, -1],
      [22, 25, 20, -1, -1, -1, -1, -1, -1, -1],
      [19, 30, 0, 11, -1, -1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ])
    lower_case_diff = [['[["i", 0, 1, "I"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'], ['[["i", 0, 1, "I"]]', '[]', '[]', '[]', '[]', '[["i", 0, 1, "I"]]', '[]', '[]', '[]', '[]'], ['[["i", 0, 1, "T"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'], ['[["i", 0, 2, "OK"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'], ['[["i", 0, 1, "H"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'], ['[["i", 0, 1, "E"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'], ['[["i", 0, 1, "U"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'], ['[["i", 0, 1, "L"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'], ['[["i", 0, 1, "L"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'], ['[["i", 0, 1, "I"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'], ['[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']]
    tokenize_diff = ['[["d", 16, 21, ""]]', '[["i", 0, 0, " "], ["d", 1, 2, ""], ["d", 23, 24, ""], ["d", 37, 38, ""]]', '[["d", 21, 22, ""], ["d", 26, 27, ""], ["d", 37, 38, ""]]', '[["i", 0, 0, " "], ["d", 2, 11, ""]]', '[["i", 0, 0, " "], ["d", 3, 4, ""], ["d", 9, 10, ""], ["d", 11, 17, ""]]', '[["d", 30, 34, ""]]', '[["i", 0, 0, " "], ["d", 14, 21, ""]]', '[["i", 0, 0, " "], ["d", 20, 27, ""]]', '[["d", 12, 19, ""]]', '[["i", 0, 0, " "], ["d", 21, 27, ""]]', '[["d", 0, 9, ""]]']
    missing_vals = np.array(['ok', 'you', 'you'], dtype='|S40')
    strings = np.array([
      [
        "It is what it is. I've seen summer and I've seen rain",
        "The sun is not yellow, it's chicken. OK. Hey, you!"
      ],
      [
        'Ended up sleeping in a doorway. Under a bodega. Lights over broadway',
        'Look out kid. Its something you did.'
      ]
    ], dtype=np.unicode)
    index_to_word = self._get_index_to_word(np.char.lower(strings), en_tokenizer)
    ids = np.array([
      'f46a7d9b0971a571b019aef7397ba54de6acf73518609b67d2ac1dee', 'f46a7d9b0971a571b019aef7397ba54de6acf73518609b67d2ac1dee', 'dd1ff31c7034df9b181dec11a0ae633bf9bee80662d76e1b5f655c2e', 'dd1ff31c7034df9b181dec11a0ae633bf9bee80662d76e1b5f655c2e', 'dd1ff31c7034df9b181dec11a0ae633bf9bee80662d76e1b5f655c2e', 'af78931ab7820443f0986de9ef1f276363014d89b9dd587f16b5f3e5', 'af78931ab7820443f0986de9ef1f276363014d89b9dd587f16b5f3e5', 'af78931ab7820443f0986de9ef1f276363014d89b9dd587f16b5f3e5', '52c66ddc2885dd3f2d30d8fbca09abab17e5d512dfd28c13f9a4bf1d', '52c66ddc2885dd3f2d30d8fbca09abab17e5d512dfd28c13f9a4bf1d', '52c66ddc2885dd3f2d30d8fbca09abab17e5d512dfd28c13f9a4bf1d'
    ])
    index_to_word = ['__UNK__'] + index_to_word[:-1]

    string_trans = st.StringTransform(
      index_to_word=index_to_word,
      word_tokenizer=en_tokenizer,
      lower_case=True,
      unk_index=0,
      max_sent_len=10,
      name='ST'
    )
    trans = dct.DocumentTransform(
      sent_tokenizer=lambda s: s.split('.'),
      sent_detokenizer=lambda s: '.'.join(s),
      string_transform=string_trans,
      keep_dims=False,
      name='DT'
    )
    # for i in xrange(2):
    self.pour_pump(
      trans,
      strings,
      {
        'DT/ST/indices': indices,
        'DT/ST/missing_vals': missing_vals,
        'DT/ST/tokenize_diff': tokenize_diff,
        'DT/ST/lower_case_diff': lower_case_diff,
        'DT/ids': ids

      },
      test_type=False
    )

    self.write_read_example(trans, strings, self.temp_dir, num_cols=1)
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
