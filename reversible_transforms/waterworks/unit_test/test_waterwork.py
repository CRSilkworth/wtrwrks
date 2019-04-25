import shutil
import tempfile
import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.transforms.num_transform as n
import reversible_transforms.tanks.add as ad
import reversible_transforms.tanks.clone as cl
import reversible_transforms.waterworks.waterwork as wa
import reversible_transforms.waterworks.globs as gl
import os
import pandas as pd
import numpy as np
import pprint

class TestWaterwork(unittest.TestCase):
  def setUp(self):
      self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
      shutil.rmtree(self.temp_dir)

  def test_pour_pump_non_eager(self):
    with wa.Waterwork() as ww:
      cl0 = cl.Clone(a=None)
      add0 = ad.Add(a=cl0['a'], b=None)
      add1 = ad.Add(a=add0['data'], b=cl0['b'])
      cl1 = cl.Clone(a=add0['a'])
      add2 = ad.Add(a=cl1['a'], b=add1['a'])

    true_funnel_dict = {
      cl0.get_slot('a'): np.array([1, 2]),
      add0.get_slot('b'): np.array([3, 4])
    }
    self.assertEqual(ww._pour_tank_order(), [cl0, add0, cl1, add1, add2])
    tap_dict = ww.pour(true_funnel_dict)

    true_tap_dict = {
        cl1['b']: np.array([1, 2]),
        add1['data']: np.array([5, 8]),
        add2['a']: np.array([1, 2]),
        add2['data']: np.array([5, 8]),
    }
    self.assertEqual(tap_dict.keys(), true_tap_dict.keys())
    for tap in tap_dict:
      th.assert_arrays_equal(self, tap_dict[tap], true_tap_dict[tap])

    self.assertEqual(ww._pump_tank_order(), [add2, add1, cl1, add0, cl0])

    funnel_dict = ww.pump(true_tap_dict)

    self.assertEqual(funnel_dict.keys(), true_funnel_dict.keys())
    for funnel in funnel_dict:
      th.assert_arrays_equal(self, funnel_dict[funnel], true_funnel_dict[funnel])

  def test_pour_pump_eager(self):
    with wa.Waterwork() as ww:
      cl0 = cl.Clone(a=np.array([1, 2]))
      add0 = ad.Add(a=cl0['a'], b=np.array([3, 4]))
      add1 = ad.Add(a=add0['data'], b=cl0['b'])
      cl1 = cl.Clone(a=add0['a'])
      add2 = ad.Add(a=cl1['a'], b=add1['a'])

    true_funnel_dict = {
      cl0.get_slot('a'): np.array([1, 2]),
      add0.get_slot('b'): np.array([3, 4])
    }
    self.assertEqual(ww._pour_tank_order(), [cl0, add0, cl1, add1, add2])
    true_tap_dict = {
        cl1['b']: np.array([1, 2]),
        add1['data']: np.array([5, 8]),
        add2['a']: np.array([1, 2]),
        add2['data']: np.array([5, 8]),
    }
    for tap in true_tap_dict:
      th.assert_arrays_equal(self, tap.get_val(), true_tap_dict[tap])

    self.assertEqual(ww._pump_tank_order(), [add2, add1, cl1, add0, cl0])

    funnel_dict = ww.pump(true_tap_dict)

    self.assertEqual(funnel_dict.keys(), true_funnel_dict.keys())
    for funnel in funnel_dict:
      th.assert_arrays_equal(self, funnel_dict[funnel], true_funnel_dict[funnel])

if __name__ == "__main__":
    unittest.main()
