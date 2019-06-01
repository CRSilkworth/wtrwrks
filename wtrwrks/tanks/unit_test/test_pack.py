import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import wtrwrks.waterworks.waterwork as wa
import numpy as np


class TestPack(th.TestTank):
  def get_lengths(self, a, default_val=0):
    zero = (a == default_val)
    all_zero = zero.all(axis=-1)
    not_zero = ~zero
    reversed_last_dim = not_zero[..., ::-1]
    lengths = np.argmax(reversed_last_dim, axis=-1)
    lengths = a.shape[-1] - lengths
    lengths[all_zero] = 0
    return lengths

  def test_two_d(self):
    a = np.array([
      [1, 1, 1, 0, 0, 0],
      [2, 2, 0, 0, 0, 0],
      [3, 3, 0, 3, 3, 0],
      [4, 0, 0, 0, 0, 0],
      [5, 5, 5, 5, 5, 5],
      [6, 6, 0, 0, 0, 0],
      [7, 7, 0, 0, 0, 0]
    ])
    target = np.array([
      [1, 1, 1, 2, 2, 0],
      [3, 3, 0, 3, 3, 4],
      [5, 5, 5, 5, 5, 5],
      [6, 6, 7, 7, 0, 0]
    ])
    ends = np.array([
      [0, 0, 1, 0, 1, 0],
      [0, 0, 0, 0, 1, 1],
      [0, 0, 0, 0, 0, 1],
      [0, 1, 0, 1, 0, 0],
    ], dtype=bool)

    self.pour_pump(
      td.pack,

      {
        'a': a,
        'default_val': 0,
        'lengths': self.get_lengths(a)
      },
      {
        'target': target,
        'default_val': 0,
        'ends': ends,
        'lengths': self.get_lengths(a)
      },
    )

    a = np.array([
      ['a', 'a', 'a', '', '', ''],
      ['b', 'b', '', '', '', ''],
      ['c1', 'c2', '', 'c3', 'c4', ''],
      ['dd', '', '', '', '', ''],
      ['e1', 'e2', 'e3', 'e4', 'e5', 'e6'],
    ])
    target = np.array([
      ['a', 'a', 'a', 'b', 'b', ''],
      ['c1', 'c2', '', 'c3', 'c4', 'dd'],
      ['e1', 'e2', 'e3', 'e4', 'e5', 'e6']
    ])
    ends = np.array([
      [0, 0, 1, 0, 1, 0],
      [0, 0, 0, 0, 1, 1],
      [0, 0, 0, 0, 0, 1],
    ], dtype=bool)

    self.pour_pump(
      td.pack,

      {
        'a': a,
        'default_val': '',
        'lengths': self.get_lengths(a, default_val='')
      },
      {
        'target': target,
        'default_val': '',
        'lengths': self.get_lengths(a, default_val=''),
        'ends': ends
      },
    )

  def test_three_d(self):
    a = np.array([
      [
        [1, 1, 1, 0, 0, 0],
        [2, 2, 0, 0, 0, 0],
        [3, 3, 0, 3, 3, 0]
      ],
      [
        [4, 0, 0, 0, 0, 0],
        [5, 5, 5, 5, 0, 0],
        [6, 6, 0, 0, 0, 0]
      ],
      [
        [7, 7, 0, 0, 0, 0],
        [8, 8, 0, 0, 0, 0],
        [9, 9, 0, 0, 0, 0]
      ],
    ])
    target = np.array([

      [
        [1, 1, 1, 2, 2, 0],
        [3, 3, 0, 3, 3, 0]
      ],
      [
        [4, 5, 5, 5, 5, 0],
        [6, 6, 0, 0, 0, 0]
      ],
      [
        [7, 7, 8, 8, 9, 9],
        [0, 0, 0, 0, 0, 0]
      ]
    ])

    ends = np.array([

      [
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
      ],
      [
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
      ],
      [
        [0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0],
      ]
    ], dtype=bool)
    is_default = a == 0

    self.pour_pump(
      td.pack,

      {
        'a': a,
        'default_val': 0,
        'lengths': self.get_lengths(a)
      },
      {
        'target': target,
        'default_val': 0,
        'lengths': self.get_lengths(a),
        'ends': ends
      },
    )

  def test_four_d(self):
    a = np.array([
      [
        [
          [1, 1, 1, 0, 0, 0],
          [2, 2, 0, 0, 0, 0]
        ],
        [
          [0, 0, 0, 0, 0, 0],
          [3, 3, 0, 3, 3, 0]
        ],
      ],
      [
        [
          [4, 0, 0, 0, 0, 0],
          [5, 5, 5, 5, 0, 0]
        ],
        [
          [6, 6, 0, 0, 0, 0],
          [7, 7, 0, 0, 0, 0]
        ]
      ],
      [
        [
          [0, 0, 0, 0, 0, 0],
          [8, 8, 0, 0, 0, 0]
        ],
        [
          [9, 9, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0]
        ]
      ],
    ])
    target = np.array([

      [
        [[1, 1, 1, 2, 2, 0]],
        [[3, 3, 0, 3, 3, 0]]
      ],
      [
        [[4, 5, 5, 5, 5, 0]],
        [[6, 6, 7, 7, 0, 0]]
      ],
      [
        [[8, 8, 0, 0, 0, 0]],
        [[9, 9, 0, 0, 0, 0]]
      ]
    ])
    ends = np.array([

      [
        [[0, 0, 1, 0, 1, 0]],
        [[0, 0, 0, 0, 1, 0]],
      ],
      [
        [[1, 0, 0, 0, 1, 0]],
        [[0, 1, 0, 1, 0, 0]],
      ],
      [
        [[0, 1, 0, 0, 0, 0]],
        [[0, 1, 0, 0, 0, 0]],
      ]
    ], dtype=bool)
    is_default = a == 0
    with wa.Waterwork() as ww:
      ts, ss = td.pack(a, self.get_lengths(a), 0)
    print ts['target'].get_val()
    self.pour_pump(
      td.pack,

      {
        'a': a,
        'default_val': 0,
        'lengths': self.get_lengths(a)
      },
      {
        'target': target,
        'default_val': 0,
        'lengths': self.get_lengths(a),
        'ends': ends
      },
    )


if __name__ == "__main__":
    unittest.main()
