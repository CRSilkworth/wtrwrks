# -*- coding: utf-8 -*-
import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np
import tinysegmenter

class TestLowerCase(th.TestTank):
  def test_scalar(self):
    strings = "It is what it is."
    diff = '[["i", 0, 1, "I"]]'
    self.pour_pump(
      td.lower_case,
      {
        'strings': strings,
      },
      {
        'target': strings.lower(),
        'diff': diff,
      },
      test_type=False
    )

  def test_one_d(self):
    strings = np.array([
      u'チラシ・勧誘印刷物の無断投函は一切お断り',
      u'すみませんが、もう一度どお願いします。'
    ])
    diff = ['[]', '[]']
    self.pour_pump(
      td.lower_case,
      {
        'strings': strings,
      },
      {
        'target': strings,
        'diff': diff,
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
      ['[["i", 0, 1, "I"]]', '[["i", 0, 1, "W"], ["i", 10, 11, "B"]]'],
      ['[["i", 0, 1, "T"], ["i", 37, 39, "OK"]]', '[["i", 0, 1, "D"]]']
    ]

    self.pour_pump(
      td.lower_case,
      {
        'strings': strings,
      },
      {
        'target': np.reshape([s.lower() for s in strings.flatten()], strings.shape),
        'diff': diff,
      },
      test_type=False
    )

if __name__ == "__main__":
    unittest.main()
