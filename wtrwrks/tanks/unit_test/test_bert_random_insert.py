# -*- coding: utf-8 -*-
import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np
import tinysegmenter

class TestBertRandomInsert(th.TestTank):

  def test_two_d(self):
    a = np.array([
      ['a', 'b', 'ccc', 'dd', 'ee', '', '', '', '', ''],
      ['a', 'b', 'c', 'dd', 'e', 'f', 'g', 'h', '', ''],
      ['aa', 'bb', 'cc', 'ddd', '', '', '', '', '', ''],
      ['a', 'bb', 'c', 'd', 'ff', 'gg', '', '', '', ''],
      ['aa', 'bb', 'cc', 'd', 'ee', '', '', '', '', ''],
      ['a', 'b', 'c', 'd', 'ee', 'f', 'gg', '', '', ''],
      ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
      ['k', 'l', 'm', 'n', 'o', 'p', 'q', 'y', 'u', 'j'],
      ['aaaaaaaaa', '', '', '', '', '', '', '', '', ''],
    ])
    segment_ids = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    ends = np.array([
      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
      [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
      [1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
      [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    is_random_next = np.array([False, True, False, True, True, True, False, False, True])
    removed = np.array([['[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]'], ['[NA]', '[NA]', '[NA]', '[NA]', 'c', 'dd', 'e', 'f', 'g', 'h', '[SEP]', '', ''], ['[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]'], ['[NA]', '[NA]', '[NA]', 'bb', 'c', 'd', 'ff', 'gg', '[SEP]', '', '', '', ''], ['[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', 'ee', '[SEP]', '', '', '', '', ''], ['[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[SEP]', '', '', ''], ['[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]'], ['[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]', '[NA]'], ['[NA]', '[NA]', '[NA]', '[SEP]', '', '', '', '', '', '', '', '', '']])

    target = np.array([
      ['[CLS]', 'a', 'b', 'ccc', 'dd', 'ee', '[SEP]', '[SEP]', '', '', '', '', ''],
      ['[CLS]', 'a', 'b', '[SEP]', 'c', 'dd', 'e', 'f', 'g', 'h', '[SEP]', '', ''],
      ['[CLS]', 'aa', 'bb', 'cc', '[SEP]', 'ddd', '[SEP]', '', '', '', '', '', ''],
      ['[CLS]', 'a', '[SEP]', 'c', 'dd', 'e', 'f', 'g', 'h', '[SEP]', '', '', ''],
      ['[CLS]', 'aa', 'bb', 'cc', 'd', '[SEP]', 'bb', 'c', 'd', 'ff', 'gg', '[SEP]', ''],
      ['[CLS]', 'a', 'b', 'c', 'd', 'ee', 'f', 'gg', '[SEP]', 'ddd', '[SEP]', '', ''],
      ['[CLS]', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', '[SEP]', '[SEP]'],
      ['[CLS]', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'y', 'u', 'j', '[SEP]', '[SEP]'],
      ['[CLS]', 'aaaaaaaaa', '[SEP]', 'c', 'dd', 'e', 'f', 'g', 'h', '[SEP]', '', '', '']
      ])

    self.pour_pump(
      td.bert_random_insert,
      {
        'a': a,
        'ends': ends,
        'num_tries': 10,
        'random_seed': 0
      },
      {
        'target': target,
        'removed': removed,
        'ends': ends,
        'num_tries': 10,
        'random_seed': 0,
        'segment_ids': segment_ids,
        'is_random_next': is_random_next
      },
    )

if __name__ == "__main__":
    unittest.main()
