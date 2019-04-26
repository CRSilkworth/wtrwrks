import shutil
import tempfile
import unittest
import reversible_transforms.utils.test_helpers as th
import reversible_transforms.tanks.add as ad
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.waterworks.waterwork as wa
import numpy as np

class TestWaterwork(unittest.TestCase):
  def setUp(self):
      self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
      shutil.rmtree(self.temp_dir)

  def test_pour_pump_non_eager(self):
    with wa.Waterwork() as ww:
      cl0 = ta.clone(a=None)
      add0 = ad.add(a=cl0['a'], b=None)
      add1 = ad.add(a=add0['target'], b=cl0['b'])
      cl1 = ta.clone(a=add0['a'])
      add2 = ad.add(a=cl1['a'], b=add1['a'])

    true_funnel_dict = {
      cl0.get_slot('a'): np.array([1, 2]),
      add0.get_slot('b'): np.array([3, 4])
    }
    self.assertEqual(ww._pour_tank_order(), [cl0, add0, cl1, add1, add2])
    tap_dict = ww.pour(true_funnel_dict)

    true_tap_dict = {
        cl1['b']: np.array([1, 2]),
        add1['target']: np.array([5, 8]),
        add2['a']: np.array([1, 2]),
        add2['target']: np.array([5, 8]),
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
      cl0 = ta.clone(a=np.array([1, 2]))
      add0 = ad.add(a=cl0['a'], b=np.array([3, 4]))
      add1 = ad.add(a=add0['target'], b=cl0['b'])
      cl1 = ta.clone(a=add0['a'])
      add2 = ad.add(a=cl1['a'], b=add1['a'])

    true_funnel_dict = {
      cl0.get_slot('a'): np.array([1, 2]),
      add0.get_slot('b'): np.array([3, 4])
    }
    self.assertEqual(ww._pour_tank_order(), [cl0, add0, cl1, add1, add2])
    true_tap_dict = {
        cl1['b']: np.array([1, 2]),
        add1['target']: np.array([5, 8]),
        add2['a']: np.array([1, 2]),
        add2['target']: np.array([5, 8]),
    }
    for tap in true_tap_dict:
      th.assert_arrays_equal(self, tap.get_val(), true_tap_dict[tap])

    self.assertEqual(ww._pump_tank_order(), [add2, add1, cl1, add0, cl0])

    funnel_dict = ww.pump(true_tap_dict)

    self.assertEqual(funnel_dict.keys(), true_funnel_dict.keys())
    for funnel in funnel_dict:
      th.assert_arrays_equal(self, funnel_dict[funnel], true_funnel_dict[funnel])

  def test_merge(self):
    with wa.Waterwork(name='ww1') as ww1:
      cl0 = ta.clone(a=np.array([1, 2]))
      add0 = ad.add(a=cl0['a'], b=np.array([3, 4]))
      add1 = ad.add(a=add0['target'], b=cl0['b'])

    with wa.Waterwork(name='ww2') as ww2:
      cl1 = ta.clone(a=None, name='ww2/Clone_1')
      add2 = ad.add(a=cl1['a'], b=None, name='ww2/Add_2')

    join_dict = {
      cl1.get_slot('a'): add0['a'],
      add2.get_slot('b'): add1['a']
    }

    ww3 = ww1.merge(ww2, join_dict, name='ww3')

    true_funnel_dict = {
      ('ww3/ww1/Clone_0', 'a'): np.array([1, 2]),
      ('ww3/ww1/Add_0', 'b'): np.array([3, 4])
    }

    tap_dict = ww3.pour(true_funnel_dict)

    true_tap_dict = {
        ('ww3/ww2/Clone_1', 'b'): np.array([1, 2]),
        ('ww3/ww1/Add_1', 'target'): np.array([5, 8]),
        ('ww3/ww2/Add_2', 'a'): np.array([1, 2]),
        ('ww3/ww2/Add_2', 'target'): np.array([5, 8]),
    }
    tap_dict = {(t.tank.name, t.key): v for t, v in tap_dict.iteritems()}

    self.assertEqual(sorted(tap_dict.keys()), sorted(true_tap_dict.keys()))
    for tap in tap_dict:
      th.assert_arrays_equal(self, tap_dict[tap], true_tap_dict[tap])

    funnel_dict = ww3.pump(true_tap_dict)

    funnel_dict = {(t.tank.name, t.key): v for t, v in funnel_dict.iteritems()}
    self.assertEqual(sorted(funnel_dict.keys()), sorted(true_funnel_dict.keys()))
    for funnel in funnel_dict:
      th.assert_arrays_equal(self, funnel_dict[funnel], true_funnel_dict[funnel])

  def test_auto_clone(self):
    def test_pour_pump_non_eager(self):
      with wa.Waterwork() as ww:
        add0 = ad.add(a=None, b=None)
        add1 = ad.add(a=add0['target'], b=None)
        add2 = ad.add(a=add0['target'], b=add1['a'])

      true_funnel_dict = {
        add0.get_slot('a'): np.array([1, 2]),
        add0.get_slot('b'): np.array([3, 4]),
        add1.get_slot('b'): np.array([3, 4])
      }
      self.assertEqual(ww._pour_tank_order(), [add0, add1, add2])
      tap_dict = ww.pour(true_funnel_dict)

      true_tap_dict = {
          add0['a']: np.array([1, 2]),
          add1['target']: np.array([6, 9]),
          add2['a']: np.array([4, 6]),
          add2['target']: np.array([8, 12]),
      }
      self.assertEqual(tap_dict.keys(), true_tap_dict.keys())
      for tap in tap_dict:
        th.assert_arrays_equal(self, tap_dict[tap], true_tap_dict[tap])

      self.assertEqual(ww._pump_tank_order(), [add2, add1, add0])

      funnel_dict = ww.pump(true_tap_dict)

      self.assertEqual(funnel_dict.keys(), true_funnel_dict.keys())
      for funnel in funnel_dict:
        th.assert_arrays_equal(self, funnel_dict[funnel], true_funnel_dict[funnel])
if __name__ == "__main__":
    unittest.main()
