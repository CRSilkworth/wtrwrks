# -*- coding: utf-8 -*-
import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import numpy as np


class TestPhaseDecomp(th.TestTank):

  def test_one_d(self):
    a = np.array([1.4, 2.0, 5.8])
    w_k = np.array([3.0, 1.4])

    target = np.array([
      [0.2, 0.96],
      [0.0, 0.8],
      [0.4, 0.12]
    ])

    div = np.array([
      [4, 1],
      [6, 2],
      [17, 8]
    ])

    self.pour_pump(
      td.phase_decomp,
      {
        'a': a,
        'w_k': w_k,
      },
      {
        'target': target,
        'div': div,
        'w_k': w_k,
      },
      test_type=False
    )

  def test_two_d(self):
    a = np.array([[1.4, 2.0, 5.8]])
    w_k = np.array([3.0, 1.4])

    target = np.array([[
      [0.2, 0.96],
      [0.0, 0.8],
      [0.4, 0.12]
    ]])

    div = np.array([[
      [4, 1],
      [6, 2],
      [17, 8]
    ]])

    self.pour_pump(
      td.phase_decomp,
      {
        'a': a,
        'w_k': w_k,
      },
      {
        'target': target,
        'div': div,
        'w_k': w_k,
      },
      test_type=False
    )

if __name__ == "__main__":
    unittest.main()
