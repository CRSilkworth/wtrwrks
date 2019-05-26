# -*- coding: utf-8 -*-
import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np
import tinysegmenter

class TestTokenize(th.TestTank):
  def test_scalar(self):
    tokenizer = lambda s: s.split()
    detokenizer = lambda a: ' '.join(a)
    strings = "It is what it is."
    diff = '[["d", 17, 22, ""]]'
    self.pour_pump(
      td.tokenize,
      {
        'strings': strings,
        'tokenizer': tokenizer,
        'max_len': 10,
        'detokenizer': detokenizer
      },
      {
        'target': ['It', 'is', 'what', 'it', 'is.', '', '', '', '', ''],
        'diff': diff,
        'tokenizer': tokenizer,
        'detokenizer': detokenizer
      },
      test_type=False
    )

  def test_one_d(self):
    ts = tinysegmenter.TinySegmenter()
    tokenizer = ts.tokenize
    detokenizer = lambda a: ''.join(a)
    strings = np.array([
      u'チラシ・勧誘印刷物の無断投函は一切お断り',
      u'すみませんが、もう一度どお願いします。'
    ])
    diff = ['[["i", 16, 16, "\\u5207\\u304a\\u65ad\\u308a"]]', '[["i", 18, 18, "\\u3002"]]']
    target = [[u'\u30c1\u30e9\u30b7', u'\u30fb', u'\u52e7\u8a98', u'\u5370\u5237', u'\u7269', u'\u306e', u'\u7121\u65ad', u'\u6295\u51fd', u'\u306f', u'\u4e00'], [u'\u3059\u307f\u307e\u305b', u'\u3093', u'\u304c', u'\u3001', u'\u3082\u3046', u'\u4e00', u'\u5ea6', u'\u3069\u304a\u9858\u3044', u'\u3057', u'\u307e\u3059']]
    self.pour_pump(
      td.tokenize,
      {
        'strings': strings,
        'tokenizer': tokenizer,
        'max_len': 10,
        'detokenizer': detokenizer
      },
      {
        'target': target,
        'diff': diff,
        'tokenizer': tokenizer,
        'detokenizer': detokenizer
      },
      test_type=False
    )

  def test_two_d(self):
    tokenizer = lambda s: s.split()
    detokenizer = lambda a: ' '.join(a)
    strings = np.array([
      [
        "It is what it is.",
        "Whatever, Bob's mother has seen the world."
      ],
      [
        "The sun is not yellow, it's chicken. OK.",
        "Don't need a weatherman to know which way the wind blows."
      ]
    ])
    diff = [
      ['[["d", 17, 22, ""]]', '[["d", 42, 45, ""]]'],
      ['[["d", 40, 42, ""]]', '[["i", 50, 50, " blows."]]']
    ]

    target = np.array(
      [
        [
          ['It', 'is', 'what', 'it', 'is.', '', '', '', '', ''], ['Whatever,', "Bob's", 'mother', 'has', 'seen', 'the', 'world.', '', '', '']
        ],
        [
          ['The', 'sun', 'is', 'not', 'yellow,', "it's", 'chicken.', 'OK.', '', ''],
          ["Don't", 'need', 'a', 'weatherman', 'to', 'know', 'which', 'way', 'the', 'wind']
        ]
      ])
    self.pour_pump(
      td.tokenize,
      {
        'strings': strings,
        'tokenizer': tokenizer,
        'max_len': 10,
        'detokenizer': detokenizer
      },
      {
        'target': target,
        'diff': diff,
        'tokenizer': tokenizer,
        'detokenizer': detokenizer
      },
      test_type=False
    )

if __name__ == "__main__":
    unittest.main()
