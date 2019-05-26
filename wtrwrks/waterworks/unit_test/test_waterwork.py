import shutil
import tempfile
import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import wtrwrks.waterworks.waterwork as wa
from wtrwrks.waterworks.empty import empty
import numpy as np
import pprint


class TestWaterwork(unittest.TestCase):
  def setUp(self):
      self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
      shutil.rmtree(self.temp_dir)

  def test_pour_pump_non_eager(self):
    with wa.Waterwork() as ww:
      add0_tubes, add0_slots = empty + empty

      add1_tubes, add1_slots = add0_tubes['target'] + empty

      cl0_tubes, _ = td.clone(a=add0_tubes['smaller_size_array'])

      add2_tubes, _ = td.add(a=cl0_tubes['a'], b=add1_tubes['smaller_size_array'])

      cl0_tubes['b'].set_name('output_1')

    true_funnel_dict = {
      add0_slots['a']: np.array([1, 2]),
      add0_slots['b']: np.array([3, 4]),
      ('Add_1', 'b'): np.array([1, 2])
    }
    # print [str(t) for t in ww._pour_tank_order()]
    self.assertEqual(ww._pour_tank_order(), [ww.tanks[k] for k in ['Add_0', 'Add_1', 'Clone_0', 'Add_2']])
    # self.assertEqual(ww._pour_tank_order(), [ww.tanks["Clone_0"], add0, cl1, add1, add2])
    tap_dict = ww.pour(true_funnel_dict)

    true_tap_dict = {
        "output_1": np.array([3, 4]),
        add0_tubes['a_is_smaller']: False,
        add1_tubes['target']: np.array([5, 8]),
        add1_tubes['a_is_smaller']: False,
        add2_tubes['smaller_size_array']: np.array([1, 2]),
        add2_tubes['a_is_smaller']: False,
        add2_tubes['target']: np.array([4, 6]),
    }
    temp_tap_dict = {}
    temp_tap_dict.update(true_tap_dict)
    temp_tap_dict[cl0_tubes['b']] = temp_tap_dict['output_1']
    del temp_tap_dict['output_1']

    # print [str(k) for k in sorted(tap_dict.keys())]
    # print [str(k) for k in sorted(temp_tap_dict.keys())]
    self.assertEqual(sorted(tap_dict.keys()), sorted(temp_tap_dict.keys()))
    for tap in tap_dict:
      th.assert_arrays_equal(self, tap_dict[tap], temp_tap_dict[tap])

    # print [str(t) for t in ww._pump_tank_order()]
    self.assertEqual(ww._pump_tank_order(), [ww.tanks[k] for k in ['Add_2', 'Add_1', 'Clone_0', 'Add_0']])
    # self.assertEqual(ww._pump_tank_order(), [add2, cl1, add1, add0, ww.tanks["Clone_0"]])

    funnel_dict = ww.pump(true_tap_dict)
    # funnel_dict = {s.get_tuple(): v for s, v in funnel_dict.iteritems()}

    temp_val = true_funnel_dict[('Add_1', 'b')]
    del true_funnel_dict[('Add_1', 'b')]
    true_funnel_dict[add1_slots['b']] = temp_val

    # print [str(k) for k in sorted(funnel_dict.keys())]
    # print [str(k) for k in sorted(true_funnel_dict.keys())]
    self.assertEqual(sorted(funnel_dict.keys()), sorted(true_funnel_dict.keys()))
    for funnel in funnel_dict:
      th.assert_arrays_equal(self, funnel_dict[funnel], true_funnel_dict[funnel])

    ww.clear_vals()
    for d in [ww.slots, ww.tubes]:
      for key in d:
        self.assertEqual(d[key].get_val(), None)

  def test_pour_pump_eager(self):
    with wa.Waterwork() as ww:
      cl0_tubes, cl0_slots = td.clone(a=np.array([1, 2]))
      add0_tubes, add0_slots = cl0_tubes['a'] + np.array([3, 4])
      add1_tubes, _ = add0_tubes['target'] + cl0_tubes['b']
      cl1_tubes, _ = td.clone(a=add0_tubes['smaller_size_array'])
      add2_tubes, _ = cl1_tubes['a'] * add1_tubes['smaller_size_array']

      add2_tubes['target'].set_name('answer')

    true_funnel_dict = {
      cl0_slots['a']: np.array([1, 2]),
      add0_slots['b']: np.array([3, 4])
    }

    # print [str(t) for t in ww._pour_tank_order()]
    # self.assertEqual(ww._pour_tank_order(), [cl0, add0, cl1, add1, add2])
    true_tap_dict = {
        cl1_tubes['b']: np.array([3, 4]),
        add1_tubes['target']: np.array([5, 8]),
        add2_tubes['smaller_size_array']: np.array([1, 2]),
        "answer": np.array([3, 8]),
    }
    temp_tap_dict = {}
    temp_tap_dict.update(true_tap_dict)
    temp_tap_dict[add2_tubes['target']] = temp_tap_dict['answer']
    del temp_tap_dict['answer']
    for tap in temp_tap_dict:
      th.assert_arrays_equal(self, tap.get_val(), temp_tap_dict[tap])

    # print [str(t) for t in ww._pump_tank_order()]
    # self.assertEqual(ww._pump_tank_order(), [add2, cl1, add1, add0, cl0])

    # print [str(k) for k in ww.taps]
    funnel_dict = ww.pump(true_tap_dict)

    self.assertEqual(sorted(funnel_dict.keys()), sorted(true_funnel_dict.keys()))
    for funnel in funnel_dict:
      th.assert_arrays_equal(self, funnel_dict[funnel], true_funnel_dict[funnel])

  def test_pour_pump_iter_tube(self):
    with wa.Waterwork() as ww:
      ls, ls_slots = td.iter_list(empty, 2)
      ds, ds_slots = td.iter_dict(ls[1], ['a', 'b'])

      add0, _ = ls[0] + ds['a']
      add1, _ = ds['b'] + add0['target']
    true_funnel_dict = {
      ls_slots['a']: [7, {'a': 4, 'b': 5}]
    }
    tap_dict = ww.pour(true_funnel_dict)

    true_tap_dict = {
      add0['smaller_size_array']: 4,
      add0['a_is_smaller']: False,
      add1['target']: 16,
      add1['smaller_size_array']: 11,
      add1['a_is_smaller']: False,
    }
    temp_tap_dict = {}
    temp_tap_dict.update(true_tap_dict)

    self.assertEqual(sorted(tap_dict.keys()), sorted(temp_tap_dict.keys()))
    for tap in tap_dict:
      th.assert_arrays_equal(self, tap_dict[tap], temp_tap_dict[tap])

    funnel_dict = ww.pump(true_tap_dict)

    self.assertEqual(sorted(funnel_dict.keys()), sorted(true_funnel_dict.keys()))
    for funnel in funnel_dict:
      self.assertEqual(funnel_dict[funnel], true_funnel_dict[funnel])

    ww.clear_vals()
    for d in [ww.slots, ww.tubes]:
      for key in d:
        self.assertEqual(d[key].get_val(), None)

  # def test_merge(self):
  #   with wa.Waterwork(name='ww1') as ww1:
  #     cl0 = td.clone(a=np.array([1, 2]))
  #     add0 = td.add(a=cl0['a'], b=np.array([3, 4]))
  #     add1 = td.add(a=add0['target'], b=cl0['b'])
  #
  #   with wa.Waterwork(name='ww2') as ww2:
  #     cl1 = td.clone(a=None, type_dict={'a': (np.ndarray, int)}, name='ww2/Clone_1')
  #     add2 = td.add(a=cl1['a'], b=None, type_dict={'b': (np.ndarray, int)}, name='ww2/Add_2')
  #
  #   join_dict = {
  #     cl1.get_slot('a'): add0['smaller_size_array'],
  #     add2.get_slot('b'): add1['smaller_size_array']
  #   }
  #
  #   ww3 = ww1.merge(ww2, join_dict, name='ww3')
  #
  #   # print [str(t) for t in ww3.funnels]
  #   true_funnel_dict = {
  #     cl0.get_slot('a'): np.array([1, 2]),
  #     add0.get_slot('b'): np.array([3, 4])
  #   }
  #
  #   tap_dict = ww3.pour(true_funnel_dict)
  #
  #   true_tap_dict = {
  #       cl1['b']: np.array([3, 4]),
  #       add0['a_is_smaller']: False,
  #       add1['target']: np.array([5, 8]),
  #       add1['a_is_smaller']: False,
  #       add2['smaller_size_array']: np.array([1, 2]),
  #       add2['target']: np.array([4, 6]),
  #       add2['a_is_smaller']: False,
  #   }
  #
  #   self.assertEqual(sorted(tap_dict.keys()), sorted(true_tap_dict.keys()))
  #   for tap in tap_dict:
  #     th.assert_arrays_equal(self, tap_dict[tap], true_tap_dict[tap])
  #
  #   funnel_dict = ww3.pump(true_tap_dict)
  #
  #   self.assertEqual(sorted(funnel_dict.keys()), sorted(true_funnel_dict.keys()))
  #   for funnel in funnel_dict:
  #     th.assert_arrays_equal(self, funnel_dict[funnel], true_funnel_dict[funnel])

if __name__ == "__main__":
    unittest.main()
