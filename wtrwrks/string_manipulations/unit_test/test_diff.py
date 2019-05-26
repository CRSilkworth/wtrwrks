import shutil
import tempfile
import unittest
import wtrwrks.string_manipulations.diff as df
import os
import pandas as pd
import numpy as np

class TestDiff(unittest.TestCase):
  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.temp_dir)

  def test_set_transform(self):
    source = "Hey Hey, what's up"
    target = "Hey hey what's happening"
    diff_string = df.get_diff_string(source, target)

    self.assertEqual(diff_string, '[["i", 4, 5, "h"], ["d", 7, 8, ""], ["i", 16, 17, "ha"], ["i", 18, 18, "pening"]]')
    reco = df.reconstruct(source, diff_string)

    self.assertEqual(target, reco)
if __name__ == "__main__":
  unittest.main()
