# -*- coding: utf-8 -*-
import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np


class TestHalfWidth(th.TestTank):

  def test_one_d(self):
    strings = np.array([
      u'１００チラシ・勧誘印刷物の無断投函は一切お断り',
      u'すみませんが、もう一度どお願いします。'
    ])
    diff = ['[["i", 0, 3, "\\uff11\\uff10\\uff10"]]', '[]']
    target = [u'100\u30c1\u30e9\u30b7\u30fb\u52e7\u8a98\u5370\u5237\u7269\u306e\u7121\u65ad\u6295\u51fd\u306f\u4e00\u5207\u304a\u65ad\u308a', u'\u3059\u307f\u307e\u305b\u3093\u304c\u3001\u3082\u3046\u4e00\u5ea6\u3069\u304a\u9858\u3044\u3057\u307e\u3059\u3002']
    self.pour_pump(
      td.half_width,
      {
        'strings': strings,
      },
      {
        'target': target,
        'diff': diff,
      },
      test_type=False
    )

  def test_two_d(self):
    strings = np.array([
      [
        u'早上好,你好吗。',
        u'我很好,你呢?８９'
      ],
      [
        u'２３チラシ・勧誘印刷物の無断投函は一切お断り',
        u'すみませんが、もう一度どお願いします。',
      ]
    ])
    # diff = [
    #   ['[["i", 0, 1, "I"]]', '[["i", 0, 1, "W"], ["i", 10, 11, "B"]]'],
    #   ['[["i", 0, 1, "T"], ["i", 37, 39, "OK"]]', '[["i", 0, 1, "D"]]']
    # ]
    diff = [['[]', '[]'], ['[]', '[]']]
    target = [
      [u'\u65e9\u4e0a\u597d,\u4f60\u597d\u5417\u3002', u'\u6211\u5f88\u597d,\u4f60\u5462?\uff18\uff19'],
      [u'\uff12\uff13\u30c1\u30e9\u30b7\u30fb\u52e7\u8a98\u5370\u5237\u7269\u306e\u7121\u65ad\u6295\u51fd\u306f\u4e00\u5207\u304a\u65ad\u308a', u'\u3059\u307f\u307e\u305b\u3093\u304c\u3001\u3082\u3046\u4e00\u5ea6\u3069\u304a\u9858\u3044\u3057\u307e\u3059\u3002']]
    self.pour_pump(
      td.lower_case,
      {
        'strings': strings,
      },
      {
        'target': target,
        'diff': diff,
      },
      test_type=False
    )

if __name__ == "__main__":
    unittest.main()
