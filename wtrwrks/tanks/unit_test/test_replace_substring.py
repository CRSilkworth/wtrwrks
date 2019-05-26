# -*- coding: utf-8 -*-
import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np
import tinysegmenter

class TestReplaceSubstring(th.TestTank):
  def test_scalar(self):
    strings = np.array("It is what it is.")
    diff = np.array('[["i", 6, 6, "what"]]')
    old_substring = 'what'
    new_substring = ''
    self.pour_pump(
      td.replace_substring,
      {
        'strings': strings,
        'old_substring': old_substring,
        'new_substring': new_substring
      },
      {
        'target': np.char.replace(strings, old_substring, new_substring),
        'diff': diff,
        'old_substring': old_substring,
        'new_substring': new_substring
      },
      test_type=False
    )

  def test_one_d(self):
    strings = np.array([
      u'チラシ・勧誘印刷物の無断投函は一切お断り',
      u'すみませんが、もう一度どお願いします。'
    ])
    diff = ['[]', '[]']
    old_substring = u'ー'
    new_substring = 'bamboozle'
    self.pour_pump(
      td.replace_substring,
      {
        'strings': strings,
        'old_substring': old_substring,
        'new_substring': new_substring
      },
      {
        'target': np.char.replace(strings, old_substring, new_substring),
        'diff': diff,
        'old_substring': old_substring,
        'new_substring': new_substring
      },
      test_type=False
    )

  def test_two_d(self):
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
      ['[["i", 3, 12, "is"], ["i", 21, 30, "is"]]', '[]'],
      ['[["i", 8, 17, "is"]]', '[]']
    ]
    old_substring = 'is'
    new_substring = 'bamboozle'
    self.pour_pump(
      td.replace_substring,
      {
        'strings': strings,
        'old_substring': old_substring,
        'new_substring': new_substring
      },
      {
        'target': np.char.replace(strings, old_substring, new_substring),
        'diff': diff,
        'old_substring': old_substring,
        'new_substring': new_substring
      },
      test_type=False
    )

if __name__ == "__main__":
    unittest.main()
