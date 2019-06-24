import shutil
import tempfile
import unittest
import wtrwrks.utils.test_helpers as th
import wtrwrks.tanks.tank_defs as td
import wtrwrks.waterworks.waterwork as wa
from wtrwrks.waterworks.empty import empty
import numpy as np
import pprint
import os

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
      ('Add_0', 'a'): np.array([1, 2]),
      ('Add_0', 'b'): np.array([3, 4]),
      ('Add_1', 'b'): np.array([1, 2])
    }
    for _ in xrange(2):
      self.assertEqual([str(t) for t in ww._pour_tank_order()], ['Add_0', 'Add_1', 'Clone_0', 'Add_2'])

      tap_dict = ww.pour(true_funnel_dict, key_type='str')

      true_tap_dict = {
          "output_1": np.array([3, 4]),
          'Add_0/tubes/a_is_smaller': False,
          'Add_1/tubes/a_is_smaller': False,
          'Add_1/tubes/target': np.array([5, 8]),
          'Add_2/tubes/a_is_smaller': False,
          'Add_2/tubes/smaller_size_array': np.array([1, 2]),
          'Add_2/tubes/a_is_smaller': False,
          'Add_2/tubes/target': np.array([4, 6]),
      }

      self.assertEqual(set(tap_dict.keys()), set(true_tap_dict.keys()))
      for tap in tap_dict:
        th.assert_arrays_equal(self, tap_dict[tap], true_tap_dict[tap])

      self.assertEqual(ww._pump_tank_order(), [ww.tanks[k] for k in ['Add_2', 'Add_1', 'Clone_0', 'Add_0']])

      funnel_dict = ww.pump(true_tap_dict, key_type='tuple')

      self.assertEqual(sorted(funnel_dict.keys()), sorted(true_funnel_dict.keys()))
      for funnel in funnel_dict:
        th.assert_arrays_equal(self, funnel_dict[funnel], true_funnel_dict[funnel])

      ww.clear_vals()
      for d in [ww.slots, ww.tubes]:
        for key in d:
          self.assertEqual(d[key].get_val(), None)
      pickle_name = os.path.join(self.temp_dir, 'ww.pickle')

      ww.save_to_file(pickle_name)
      ww = wa.Waterwork(from_file=pickle_name)

  def test_pour_pump_eager(self):
    with wa.Waterwork() as ww:
      cl0_tubes, cl0_slots = td.clone(a=np.array([1, 2]))
      cl0_slots['a'].unplug()

      add0_tubes, add0_slots = cl0_tubes['a'] + np.array([3, 4])
      add0_slots['b'].unplug()

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

  def test_plug(self):
    with wa.Waterwork() as ww:
      add0_tubes, add0_slots = empty + empty

      add0_slots['b'].set_plug(np.array([3, 4]))
      add0_tubes['a_is_smaller'].set_plug(False)

      add1_tubes, add1_slots = add0_tubes['target'] + empty

      add1_slots['b'].set_plug(
        lambda d: 0.5 * d['Add_0/slots/a'] + np.array([0.5, 1.0])
      )
      add1_tubes['a_is_smaller'].set_plug(
        lambda d: not d['output_1'].any()
      )

      cl0_tubes, _ = td.clone(a=add0_tubes['smaller_size_array'])

      add2_tubes, _ = td.add(a=cl0_tubes['a'], b=add1_tubes['smaller_size_array'])

      add2_tubes['target'].set_plug(
        lambda d: d['output_1'] + np.array([1, 2])
      )

      cl0_tubes['b'].set_name('output_1')

    true_funnel_dict = {
      ('Add_0', 'a'): np.array([1, 2])
    }
    for _ in xrange(2):
      self.assertEqual([str(t) for t in ww._pour_tank_order()], ['Add_0', 'Add_1', 'Clone_0', 'Add_2'])

      tap_dict = ww.pour(true_funnel_dict, key_type='str')

      true_tap_dict = {
          "output_1": np.array([3, 4]),
          'Add_1/tubes/target': np.array([5, 8]),
          'Add_2/tubes/a_is_smaller': False,
          'Add_2/tubes/smaller_size_array': np.array([1, 2]),
          'Add_2/tubes/a_is_smaller': False,
          # 'Add_2/tubes/target': np.array([4, 6]),
      }

      self.assertEqual(set(tap_dict.keys()), set(true_tap_dict.keys()))
      for tap in tap_dict:
        th.assert_arrays_equal(self, tap_dict[tap], true_tap_dict[tap])

      self.assertEqual(ww._pump_tank_order(), [ww.tanks[k] for k in ['Add_2', 'Add_1', 'Clone_0', 'Add_0']])

      funnel_dict = ww.pump(true_tap_dict, key_type='tuple')

      self.assertEqual(sorted(funnel_dict.keys()), sorted(true_funnel_dict.keys()))
      for funnel in funnel_dict:
        th.assert_arrays_equal(self, funnel_dict[funnel], true_funnel_dict[funnel])

      ww.clear_vals()
      for d in [ww.slots, ww.tubes]:
        for key in d:
          self.assertEqual(d[key].get_val(), None)
      pickle_name = os.path.join(self.temp_dir, 'ww.pickle')

      ww.save_to_file(pickle_name)
      ww = wa.Waterwork(from_file=pickle_name)

  def test_list(self):
    with wa.Waterwork() as ww:
      add0_tubes, add0_slots = empty + empty

      add0_slots['b'].set_plug(2)
      add0_tubes['a_is_smaller'].set_plug(False)

      reshape = td.reshape(empty, [1, add0_tubes['smaller_size_array']])
    true_funnel_dict = {
      ('Add_0', 'a'): np.array([1, 2]),
      ('Reshape_0', 'a'): np.array([[[3, 4]]])
    }
    for _ in xrange(2):
      tap_dict = ww.pour(true_funnel_dict, key_type='str')

      true_tap_dict = {
        'Reshape_0/tubes/target': np.array([[3, 4]]),
        'Reshape_0/tubes/old_shape': [1, 1, 2],
        'Add_0/tubes/target': np.array([3, 4]),
      }

      self.assertEqual(set(tap_dict.keys()), set(true_tap_dict.keys()))
      for tap in tap_dict:
        th.assert_arrays_equal(self, tap_dict[tap], true_tap_dict[tap])

      funnel_dict = ww.pump(true_tap_dict, key_type='tuple')

      self.assertEqual(sorted(funnel_dict.keys()), sorted(true_funnel_dict.keys()))
      for funnel in funnel_dict:
        th.assert_arrays_equal(self, funnel_dict[funnel], true_funnel_dict[funnel])

      ww.clear_vals()
      for d in [ww.slots, ww.tubes]:
        for key in d:
          self.assertEqual(d[key].get_val(), None)
      pickle_name = os.path.join(self.temp_dir, 'ww.pickle')

      ww.save_to_file(pickle_name)
      ww = wa.Waterwork(from_file=pickle_name)

  def test_multi(self):
    with wa.Waterwork() as ww:
      add0_tubes, add0_slots = empty + empty

      add0_slots['b'].set_plug(np.array([3, 4]))
      add0_tubes['a_is_smaller'].set_plug(False)

      add1_tubes, add1_slots = add0_tubes['target'] + empty

      add1_slots['b'].set_plug(
        lambda d: 0.5 * d['Add_0/slots/a'] + np.array([0.5, 1.0])
      )
      add1_tubes['a_is_smaller'].set_plug(
        lambda d: not d['output_1'].any()
      )

      cl0_tubes, _ = td.clone(a=add0_tubes['smaller_size_array'])

      add2_tubes, _ = td.add(a=cl0_tubes['a'], b=add1_tubes['smaller_size_array'])

      add2_tubes['target'].set_plug(
        lambda d: d['output_1'] + np.array([1, 2])
      )

      cl0_tubes['b'].set_name('output_1')

    true_funnel_dict = {
      ('Add_0', 'a'): np.array([1, 2])
    }
    funnel_dicts = [true_funnel_dict] * 3
    tap_dicts = ww.multi_pour(funnel_dicts, key_type='str')
    for tap_dict in tap_dicts:
      # tap_dict = ww.pour(true_funnel_dict, key_type='str')

      true_tap_dict = {
          "output_1": np.array([3, 4]),
          'Add_1/tubes/target': np.array([5, 8]),
          'Add_2/tubes/a_is_smaller': False,
          'Add_2/tubes/smaller_size_array': np.array([1, 2]),
          'Add_2/tubes/a_is_smaller': False,
          # 'Add_2/tubes/target': np.array([4, 6]),
      }

      self.assertEqual(set(tap_dict.keys()), set(true_tap_dict.keys()))
      for tap in tap_dict:
        th.assert_arrays_equal(self, tap_dict[tap], true_tap_dict[tap])

      self.assertEqual(ww._pump_tank_order(), [ww.tanks[k] for k in ['Add_2', 'Add_1', 'Clone_0', 'Add_0']])

    funnel_dicts = ww.multi_pump(tap_dicts, key_type='tuple')
    for funnel_dict in funnel_dicts:
      self.assertEqual(sorted(funnel_dict.keys()), sorted(true_funnel_dict.keys()))
      for funnel in funnel_dict:
        th.assert_arrays_equal(self, funnel_dict[funnel], true_funnel_dict[funnel])

      ww.clear_vals()
      for d in [ww.slots, ww.tubes]:
        for key in d:
          self.assertEqual(d[key].get_val(), None)
      pickle_name = os.path.join(self.temp_dir, 'ww.pickle')


if __name__ == "__main__":
    unittest.main()
