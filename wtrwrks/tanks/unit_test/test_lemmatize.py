# -*- coding: utf-8 -*-
import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np
import tinysegmenter
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

word_net_lemmatizer = WordNetLemmatizer()


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


class TestLemmatize(th.TestTank):
  def test_scalar(self):
    strings = "running"
    diff = '[["i", 3, 3, "ning"]]'
    self.pour_pump(
      td.lemmatize,
      {
        'strings': strings,
        'lemmatizer': basic_lemmatizer,
      },
      {
        'target': np.array(u'run'),
        'diff': diff,
        'lemmatizer': basic_lemmatizer,
      },
      test_type=False
    )

  def test_one_d(self):
    lemmatizer = lambda s: s
    diff = [['[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'], ['[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']]
    strings = np.array([[u'\u30c1\u30e9\u30b7', u'\u30fb', u'\u52e7\u8a98', u'\u5370\u5237', u'\u7269', u'\u306e', u'\u7121\u65ad', u'\u6295\u51fd', u'\u306f', u'\u4e00'], [u'\u3059\u307f\u307e\u305b', u'\u3093', u'\u304c', u'\u3001', u'\u3082\u3046', u'\u4e00', u'\u5ea6', u'\u3069\u304a\u9858\u3044', u'\u3057', u'\u307e\u3059']], dtype=np.unicode)
    target = np.array([[u'\u30c1\u30e9\u30b7', u'\u30fb', u'\u52e7\u8a98', u'\u5370\u5237', u'\u7269', u'\u306e', u'\u7121\u65ad', u'\u6295\u51fd', u'\u306f', u'\u4e00'], [u'\u3059\u307f\u307e\u305b', u'\u3093', u'\u304c', u'\u3001', u'\u3082\u3046', u'\u4e00', u'\u5ea6', u'\u3069\u304a\u9858\u3044', u'\u3057', u'\u307e\u3059']], dtype=np.unicode)
    self.pour_pump(
      td.lemmatize,
      {
        'strings': strings,
        'lemmatizer': lemmatizer,
      },
      {
        'target': target,
        'diff': diff,
        'lemmatizer': lemmatizer,
      },
      test_type=False
    )

  def test_two_d(self):

    diff = [
      [
        ['[]', '[["i", 0, 2, "is"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'],
        ['[]', '[]', '[]', '[["i", 2, 4, "s"]]', '[["i", 3, 3, "n"]]', '[]', '[]', '[]', '[]', '[]']],
      [
        ['[]', '[]', '[["i", 0, 2, "is"]]', '[]', '[]', '[]', '[]', '[]', '[]', '[]'],
        ['[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]', '[]']
      ]
    ]

    strings = np.array(
      [
        [
          ['It', 'is', 'what', 'it', 'is.', '', '', '', '', ''], ['Whatever,', "Bob's", 'mother', 'has', 'seen', 'the', 'world.', '', '', '']
        ],
        [
          ['The', 'sun', 'is', 'not', 'yellow,', "it's", 'chicken.', 'OK.', '', ''],
          ["Don't", 'need', 'a', 'weatherman', 'to', 'know', 'which', 'way', 'the', 'wind']
        ]
      ])
    target = np.array([[['It', 'be', 'what', 'it', 'is.', '', '', '', '', ''], ['Whatever,', "Bob's", 'mother', 'have', 'see', 'the', 'world.', '', '', '']], [['The', 'sun', 'be', 'not', 'yellow,', "it's", 'chicken.', 'OK.', '', ''], ["Don't", 'need', 'a', 'weatherman', 'to', 'know', 'which', 'way', 'the', 'wind']]])
    self.pour_pump(
      td.lemmatize,
      {
        'strings': strings,
        'lemmatizer': basic_lemmatizer,
      },
      {
        'target': target,
        'diff': diff,
        'lemmatizer': basic_lemmatizer,
      },
      test_type=False
    )

if __name__ == "__main__":
    unittest.main()
