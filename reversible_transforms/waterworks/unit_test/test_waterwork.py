import shutil
import tempfile
import unittest
import reversible_transforms.utils.test_helpers as th
# import reversible_transforms.tanks.add as ad
# import reversible_transforms.tanks.clone as cl
import reversible_transforms.tanks.tank_defs as td
import reversible_transforms.waterworks.waterwork as wa
import reversible_transforms.waterworks.placeholder as pl
import numpy as np
import pprint


class TestWaterwork(unittest.TestCase):
  def setUp(self):
      self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
      shutil.rmtree(self.temp_dir)

  def test_pour_pump_non_eager(self):
    with wa.Waterwork() as ww:
      ph0 = pl.placeholder(np.ndarray, int)
      ph1 = pl.placeholder(np.ndarray, int, name='ph1')

      add0 = ph0 + ph1

      add1 = add0['target'] + ph0

      cl1 = td.clone(a=add0['smaller_size_array'])

      add2 = td.add(a=cl1['a'], b=add1['smaller_size_array'])

      cl1['b'].set_name('output_1')

    ph_dict = {
      ph0: np.array([1, 2]),
      "ph1": np.array([3, 4])
    }
    true_funnel_dict = {
      ("CloneTyped_0", 'a'): np.array([1, 2]),
      ("AddTyped_0", 'b'): np.array([3, 4])
    }

    # print [str(t) for t in ww._pour_tank_order()]
    self.assertEqual(ww._pour_tank_order(), [ww.tanks["CloneTyped_0"], add0, cl1, add1, add2])
    tap_dict = ww.pour(ph_dict)

    true_tap_dict = {
        "output_1": np.array([3, 4]),
        add0['a_is_smaller']: False,
        add1['target']: np.array([5, 8]),
        add1['a_is_smaller']: False,
        add2['smaller_size_array']: np.array([1, 2]),
        add2['a_is_smaller']: False,
        add2['target']: np.array([4, 6]),
    }
    temp_tap_dict = {}
    temp_tap_dict.update(true_tap_dict)
    temp_tap_dict[cl1['b']] = temp_tap_dict['output_1']
    del temp_tap_dict['output_1']

    self.assertEqual(sorted(tap_dict.keys()), sorted(temp_tap_dict.keys()))
    for tap in tap_dict:
      th.assert_arrays_equal(self, tap_dict[tap], temp_tap_dict[tap])

    # print [str(t) for t in ww._pump_tank_order()]
    self.assertEqual(ww._pump_tank_order(), [add2, cl1, add1, add0, ww.tanks["CloneTyped_0"]])

    funnel_dict = ww.pump(true_tap_dict)
    funnel_dict = {s.get_tuple(): v for s, v in funnel_dict.iteritems()}

    self.assertEqual(sorted(funnel_dict.keys()), sorted(true_funnel_dict.keys()))
    for funnel in funnel_dict:
      th.assert_arrays_equal(self, funnel_dict[funnel], true_funnel_dict[funnel])

    ww.clear_vals()
    for d in [ww.slots, ww.tubes, ww.placeholders]:
      for key in d:
        self.assertEqual(d[key].get_val(), None)

  def test_pour_pump_eager(self):
    with wa.Waterwork() as ww:
      cl0 = td.clone(a=np.array([1, 2]))
      add0 = cl0['a'] + np.array([3, 4])
      add1 = add0['target'] + cl0['b']
      cl1 = td.clone(a=add0['smaller_size_array'])
      add2 = cl1['a'] * add1['smaller_size_array']

      add2['target'].set_name('answer')

    true_funnel_dict = {
      cl0.get_slot('a'): np.array([1, 2]),
      add0.get_slot('b'): np.array([3, 4])
    }

    # print [str(t) for t in ww._pour_tank_order()]
    self.assertEqual(ww._pour_tank_order(), [cl0, add0, cl1, add1, add2])
    true_tap_dict = {
        cl1['b']: np.array([3, 4]),
        add1['target']: np.array([5, 8]),
        add2['smaller_size_array']: np.array([1, 2]),
        "answer": np.array([3, 8]),
    }
    temp_tap_dict = {}
    temp_tap_dict.update(true_tap_dict)
    temp_tap_dict[add2['target']] = temp_tap_dict['answer']
    del temp_tap_dict['answer']
    for tap in temp_tap_dict:
      th.assert_arrays_equal(self, tap.get_val(), temp_tap_dict[tap])

    # print [str(t) for t in ww._pump_tank_order()]
    self.assertEqual(ww._pump_tank_order(), [add2, cl1, add1, add0, cl0])

    funnel_dict = ww.pump(true_tap_dict)

    self.assertEqual(sorted(funnel_dict.keys()), sorted(true_funnel_dict.keys()))
    for funnel in funnel_dict:
      th.assert_arrays_equal(self, funnel_dict[funnel], true_funnel_dict[funnel])

  def test_merge(self):
    with wa.Waterwork(name='ww1') as ww1:
      cl0 = td.clone(a=np.array([1, 2]))
      add0 = td.add(a=cl0['a'], b=np.array([3, 4]))
      add1 = td.add(a=add0['target'], b=cl0['b'])

    with wa.Waterwork(name='ww2') as ww2:
      cl1 = td.clone(a=None, type_dict={'a': (np.ndarray, int)}, name='ww2/Clone_1')
      add2 = td.add(a=cl1['a'], b=None, type_dict={'b': (np.ndarray, int)}, name='ww2/Add_2')

    join_dict = {
      cl1.get_slot('a'): add0['smaller_size_array'],
      add2.get_slot('b'): add1['smaller_size_array']
    }

    ww3 = ww1.merge(ww2, join_dict, name='ww3')

    # print [str(t) for t in ww3.funnels]
    true_funnel_dict = {
      cl0.get_slot('a'): np.array([1, 2]),
      add0.get_slot('b'): np.array([3, 4])
    }

    tap_dict = ww3.pour(true_funnel_dict)

    true_tap_dict = {
        cl1['b']: np.array([3, 4]),
        add0['a_is_smaller']: False,
        add1['target']: np.array([5, 8]),
        add1['a_is_smaller']: False,
        add2['smaller_size_array']: np.array([1, 2]),
        add2['target']: np.array([4, 6]),
        add2['a_is_smaller']: False,
    }

    self.assertEqual(sorted(tap_dict.keys()), sorted(true_tap_dict.keys()))
    for tap in tap_dict:
      th.assert_arrays_equal(self, tap_dict[tap], true_tap_dict[tap])

    funnel_dict = ww3.pump(true_tap_dict)

    self.assertEqual(sorted(funnel_dict.keys()), sorted(true_funnel_dict.keys()))
    for funnel in funnel_dict:
      th.assert_arrays_equal(self, funnel_dict[funnel], true_funnel_dict[funnel])

if __name__ == "__main__":
    unittest.main()
