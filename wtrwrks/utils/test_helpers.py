import numpy as np
import tensorflow as tf
import os
import wtrwrks.utils.dir_functions as d
import imp
import inspect as i
import datetime
import unittest
import shutil
import tempfile
import wtrwrks.waterworks.waterwork as wa
import pprint
import pandas as pd


class WWTest(unittest.TestCase):
  def setUp(self):
      self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
      shutil.rmtree(self.temp_dir)

  def equals(self, first, second, test_type=True):

    if type(first) in (np.dtype, type(int)):
      pass
    elif type(first) is list:
      self.assertEqual(len(first), len(second))
      for f, s in zip(first, second):
        try:
          if type(f) is np.ndarray:
            self._arrays_equal(f, s)
          elif type(f) is float and np.isnan(f):
            self.assertTrue(np.isnan(s))
          else:
            self.assertEqual(f, s)
        except(ValueError, AttributeError, AssertionError) as e:
          print "FIRST", f, type(f)
          print "SECOND", s, type(s)
          raise e
    elif type(first) is not np.ndarray:
      try:
        self.assertEqual(first, second)
        if test_type:
          self.assertTrue(type(first) is type(second))
      except (ValueError, AssertionError) as e:
        print "FIRST", first, type(first)
        print "SECOND", second, type(second)
        raise e
    else:
      self._arrays_equal(first, second, test_type=test_type)

  def _arrays_equal(self, first, second, test_type=True):
    try:
      self.assertTrue(arrays_equal(first, second, test_type=test_type))
    except AssertionError as e:
      print 'SHAPES', first.shape, np.array(second).shape
      print "FIRST ", first.tolist(), first.dtype
      print "SECOND", second, np.array(second).dtype
      if first.shape == np.array(second).shape:
        mask = first == second
        print 'DIFF', first[~mask], second[~mask]
      raise e


class TestTank (WWTest):
  def pour_pump(self, tank_func, input_dict, output_dict, type_dict=None, test_type=True):
    with wa.Waterwork() as ww:
      # test eager
      tank_tubes, tank_slots = tank_func(**input_dict)

      for key in input_dict:
        tank_slots[key].unplug()

      tank = tank_tubes[tank_tubes.keys()[0]].tank
      out_dict = {t: v.val for t, v in tank_tubes.iteritems()}
      self.assertEqual(sorted(out_dict.keys()), sorted(output_dict.keys()))
      for key in out_dict:
        try:
          self.equals(out_dict[key], output_dict[key], test_type=test_type)
        # except (ValueError, AssertionError) as e:
        except (AssertionError) as e:
          print 'Pour direction, key:', key
          raise e

    # test pour
    out_dict = tank.pour(**input_dict)
    out_dict = {t: v for t, v in out_dict.iteritems()}

    self.assertEqual(sorted(out_dict.keys()), sorted(output_dict.keys()))
    for key in out_dict:
      try:
        self.equals(out_dict[key], output_dict[key], test_type=test_type)
      except (ValueError, AssertionError) as e:
        print 'Pour direction, key:', key
        raise e

    in_dict = tank.pump(**out_dict)

    self.assertEqual(sorted(in_dict.keys()), sorted(input_dict.keys()))
    for key in in_dict:
      try:
        self.equals(in_dict[key], input_dict[key], test_type=test_type)
      except (ValueError, AssertionError) as e:
        print 'Pump direction, key:', key
        raise e


class TestTransform(WWTest):
  def pour_pump(self, trans, array, output_dict, test_type=True):
    # trans.calc_global_values(array)
    tap_dict = trans.pour(array)
    out_dict = {str(k): v for k, v in tap_dict.iteritems()}

    self.assertEqual(sorted(out_dict.keys()), sorted(output_dict.keys()))
    for key in out_dict:
      try:
        self.equals(out_dict[key], output_dict[key], test_type=test_type)
      except (ValueError, AssertionError) as e:
        print 'Pour direction, key:', key
        raise e

    original = trans.pump(out_dict)
    try:
      self.equals(original, array, test_type=test_type)
    except (ValueError, AssertionError) as e:
      raise e

  def write_read(self, trans, temp_dir):
    temp_file_path = os.path.join(temp_dir, 'temp.pickle')
    trans.save_to_file(temp_file_path)
    cls = trans.__class__
    trans = cls(from_file=temp_file_path)

    return trans

  def write_read_example(self, trans, array, dir, test_type=True, num_cols=None):
    remade_array = None
    tap_dict = trans.pour(array)
    example_dicts = trans.tap_dict_to_examples(tap_dict)
    file_name = os.path.join(dir, 'temp.tfrecord')
    writer = tf.python_io.TFRecordWriter(file_name)

    for feature_dict in example_dicts:
      example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict)
      )
      writer.write(example.SerializeToString())

    writer.close()

    dataset = tf.data.TFRecordDataset(file_name)

    if num_cols is None:
      num_cols = array.shape[1]

    dataset = dataset.map(trans.read_and_decode)
    iter = tf.data.Iterator.from_structure(
      dataset.output_types,
      dataset.output_shapes
    )
    init = iter.make_initializer(dataset)
    features = iter.get_next()

    for key in features:
      features[key] = tf.reshape(features[key], [-1])
    with tf.Session() as sess:
      sess.run(init)

      example_dicts = []
      try:
        while True:
          example_dict = sess.run(features)
          example_dicts.append(example_dict)
      except tf.errors.OutOfRangeError:
        pass
    tap_dict = trans.examples_to_tap_dict(example_dicts)
    remade_array = trans.pump(tap_dict)
    if type(array) is pd.DataFrame:
      array = array.values
    self.equals(array, remade_array, test_type)

class TestDataset (TestTransform):
  def pour_pump(self, dt, array, output_dict, test_type=True):
    # dt.calc_global_values(array)
    tap_dict = dt.pour(array)

    out_dict = {str(k): v for k, v in tap_dict.iteritems()}

    self.assertEqual(sorted(out_dict.keys()), sorted(output_dict.keys()))
    for key in out_dict:
      try:
        self.equals(out_dict[key], output_dict[key], test_type=test_type)
      except (ValueError, AssertionError) as e:
        raise e

    original = dt.pump(out_dict)

    try:
      self.equals(original, array, test_type=test_type)
    except (ValueError, AssertionError) as e:
      raise e


def arrays_equal(first, second, threshold=0.001, test_type=True):
  first = np.array(first, copy=True)
  second = np.array(second, copy=True)

  # Check that the arrays are the same shape.
  if first.shape != second.shape:
    return False
  if test_type and first.dtype != second.dtype:
    return False
  if first.size == 0 and second.size == 0:
    return True

  if np.issubdtype(first.dtype, np.datetime64) or np.issubdtype(first.dtype, np.timedelta64):
    if not (np.isnat(first) == np.isnat(second)).all():
      return False
    if np.issubdtype(first.dtype, np.datetime64):
      default = datetime.datetime(1970, 1, 1)
    else:
      default = datetime.timedelta(0)

    first[np.isnat(first)] = default
    second[np.isnat(second)] = default

    return (first == second).all()
  elif not np.issubdtype(first.dtype, np.number) and not np.issubdtype(second.dtype, np.number):
    # print '-'*20
    # mask = first.astype(np.unicode) != second.astype(np.unicode)
    # print first[mask], second[mask]
    # print '-'*20
    return (first.astype(np.unicode) == second.astype(np.unicode)).all()
  try:
    if not (np.isnan(first.astype(np.float64)) == np.isnan(second.astype(np.float64))).all():
      return False

    if not (np.isneginf(first.astype(np.float64)) == np.isneginf(second.astype(np.float64))).all():
      return False

    if not (np.isposinf(first.astype(np.float64)) == np.isposinf(second.astype(np.float64))).all():
      return False

    first[np.isnan(first.astype(np.float64))] = 0.0
    second[np.isnan(second.astype(np.float64))] = 0.0

    first[np.isneginf(first.astype(np.float64))] = 0.0
    second[np.isneginf(second.astype(np.float64))] = 0.0

    first[np.isposinf(first.astype(np.float64))] = 0.0
    second[np.isposinf(second.astype(np.float64))] = 0.0
  except ValueError:
    pass
  if threshold is None or not np.issubdtype(first.dtype, np.number):
    return (first == second).all()
  else:
    return (np.abs(first - second) < threshold).all()


def assert_arrays_equal(test_obj, a1, a2, threshold=None):
  """Shortcut for testing whether two numpy arrays are close enough to equal. Prints out the arrays and throws a unittest exception if they aren't.

  Parameters
  ----------
  test_obj : unittest subclass instance
    The unittest subclass instance to send the exceptions to.
  a1 : np.array
    First array
  a2 : np.array
    Second array
  threshold : float
    The allowed difference in values in a1 and a2 to still be considered equal

  """
  a1 = np.array(a1, copy=True)
  a2 = np.array(a2, copy=True)
  # Check that the arrays are the same shape.
  test_obj.assertEqual(a1.shape, a2.shape)

  if np.issubdtype(a1.dtype, np.datetime64) or np.issubdtype(a1.dtype, np.timedelta64):
    test_obj.assertTrue(
      (np.isnat(a1) == np.isnat(a2)).all()
    )
    a1[np.isnat(a1)] = datetime.datetime(1970, 1, 1)
    a2[np.isnat(a2)] = datetime.datetime(1970, 1, 1)

  try:
    test_obj.assertTrue(
      (np.isnan(a1.astype(np.float64)) == np.isnan(a2.astype(np.float64))).all()
    )
    a1[np.isnan(a1.astype(np.float64))] = 0.0
    a2[np.isnan(a2.astype(np.float64))] = 0.0
  except ValueError:
    pass

  # Catch the exception so the arrays can be printed first.
  try:
    # If no threshold is given then require exact match. Otherwise, test if
    # the values are just within some threshold of each other.

    if threshold is None:
      test_obj.assertTrue((a1 == a2).all())
    else:
      test_obj.assertTrue((np.abs(a1 - a2) < threshold).all())

  except Exception as e:
    # Print the arrays then raise the orginal exception.
    print a1
    print a2
    print a1.dtype
    raise e


def assert_tensor_equal_array(test_obj, a1, a2, threshold=None):
  """Shortcut for testing whether the evaled tensor is equal to some array. It will create the session, etc. so that does not need to be done outside this function.

  Parameters
  ----------
  test_obj : unittest subclass instance
    The unittest subclass instance to send the exceptions to.
  a1 : tf.tensor
    The tensor to be evaled then tested against the second array
  a2 : np.array
    Second array
  threshold : float
    The allowed difference in values in a1 and a2 to still be considered equal

  """
  # Initialize any variables that may exist in the graph
  init_op = tf.group(
    tf.global_variables_initializer(),
    tf.local_variables_initializer()
  )

  # Run the session to eval the tensor.
  with tf.Session() as sess:
    sess.run(init_op)
    a = sess.run(a1)

  # Check that the arrays are the same shape.
  test_obj.assertEqual(a.shape, a2.shape)

  # Catch the exception so the arrays can be printed first.
  try:
    # If no threshold is given then require exact match. Otherwise, test if
    # the values are just within some threshold of each other.
    if threshold is None:
      test_obj.assertTrue((a == a2).all())
    else:
      test_obj.assertTrue((np.abs(a - a2) < threshold).all())

  except Exception as e:
    # Print the arrays then raise the orginal exception.
    print a
    print a2
    raise e


def print_tensor(a1, to_list=False):
  """Shortcut for testing whether the evaled tensor is equal to some array. It will create the session, etc. so that does not need to be done outside this function.

  Parameters
  ----------
  test_obj : unittest subclass instance
    The unittest subclass instance to send the exceptions to.
  a1 : tf.tensor
    The tensor to be evaled then tested against the second array
  a2 : np.array
    Second array
  threshold : float
    The allowed difference in values in a1 and a2 to still be considered equal

  """
  # Initialize any variables that may exist in the graph
  init_op = tf.group(
    tf.global_variables_initializer(),
    tf.local_variables_initializer()
  )

  # Run the session to eval the tensor.
  with tf.Session() as sess:
    sess.run(init_op)
    a = sess.run(a1)
  if to_list:
    a = a.tolist()
  print a


def create_test_skeleton(file_name, class_name, module_key):
  """Creates a skeleton of the unittest of some particular module with a lot of the standard information already filled in.

  Parameters
  ----------
  file_name : str
    The file name of the module to be unittested
  class_name : str
    Name of the unittest class (usually just Test<Camel case version of file_name>)
  module_key : the alias of the module (usually just a two letter shortening of the file_name)
    Description of parameter `module_key`.

  Returns
  -------
  type
    Description of returned object.

  """

  # Define the template unittest file
  template = """import <OBJ_MOD> as o
import shutil
import tempfile
import unittest
import production.utils.test_helpers as th
import <MOD_PATH> as <KEY>
import os
import production.utils.dir_functions as d

this_path = os.path.dirname(os.path.realpath(__file__))

class <CLASS_NAME>(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  <TEST_FUNCTIONS>

if __name__ == "__main__":
  unittest.main()"""

  obj_mod_file_name = file_name.replace('.py', '_objects.py')
  obj_mod = obj_mod_file_name.replace('.py', '')

  unit_test_dir = d.maybe_create_dir('unit_test')
  with open(os.path.join(unit_test_dir, obj_mod_file_name), 'w') as obj_mod_file:
    obj_mod_file.write('')

  mod_path = os.getcwd()
  for pythonpath in os.environ['PYTHONPATH'].split(':'):
    mod_path = mod_path.replace(pythonpath, '')

  mod_path = mod_path.replace('/', '.').replace('.py', '').lstrip('.')
  mod_path += '.' + file_name.replace('.py', '')
  loaded = imp.load_source('*', os.path.join(os.getcwd(), file_name))

  func_tuples = []
  func_strs = []
  func_tuples.extend(i.getmembers(loaded, predicate=i.isfunction))

  for cls in i.getmembers(loaded, predicate=i.isclass):
    func_tuples.extend(i.getmembers(cls, predicate=i.ismethod))

  for func_name, func in func_tuples:
    n_args = func.__code__.co_argcount
    func_str = 'def test_' + func_name + '(self):'
    func_str += '\n    outputs = ' + module_key + '.' + func_name + '('
    func_str += '\n      ' + ',\n      '.join(func.__code__.co_varnames[:n_args])
    func_str += '\n    )'
    func_strs.append(func_str)

  test_func_str = '\n\n  '.join(func_strs)

  template = template.replace('<OBJ_MOD>', obj_mod)
  template = template.replace('<MOD_PATH>', mod_path)
  template = template.replace('<KEY>', module_key)
  template = template.replace('<CLASS_NAME>', class_name)
  template = template.replace('<TEST_FUNCTIONS>', test_func_str)

  test_file_name = 'test_' + file_name
  with open(os.path.join(unit_test_dir, test_file_name), 'w') as test_file:
    test_file.write(template)
