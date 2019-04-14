import numpy as np
import tensorflow as tf
import os
import reversible_transforms.utils.dir_functions as d
import imp
import inspect as i


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

  # Catch the exception so the arrays can be printed first.
  try:
    # If no threshold is given then require exact match. Otherwise, test if
    # the values are just within some threshold of each other.
    try:
      test_obj.assertTrue(
        (np.isnan(a1.astype(np.float64)) == np.isnan(a2.astype(np.float64))).all()
      )
      a1[np.isnan(a1.astype(np.float64))] = 0.0
      a2[np.isnan(a2.astype(np.float64))] = 0.0
    except ValueError:
      pass
    if threshold is None:
      test_obj.assertTrue((a1 == a2).all())
    else:
      test_obj.assertTrue((np.abs(a1 - a2) < threshold).all())

  except Exception as e:
    # Print the arrays then raise the orginal exception.
    print a1
    print a2
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


if __name__ == "__main__":
  create_test_skeleton('create_data.py', 'TestCreateData', 'cr')
