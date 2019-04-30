import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import numpy as np


def create_one_arg_bool_tank(np_func, class_name):
  """Create a function which generates the tank instance corresponding to some single argument, boolean valued numpy function. (e.g. np.isnan). The operation will be reversible but in the most trivial and wasteful manner possible. It will just copy over the original array.

  Parameters
  ----------
  np_func : numpy function
      A numpy function which operates on an array to give another array.
  class_name : str
      The name you'd like to have the Tank class called.

  Returns
  -------
  func
      A function which outputs a tank instance which behaves like the np_func but is also reversible

  """
  # Define the tank subclass.
  class TankClass(ta.Tank):

    slot_keys = ['a']
    tube_dict = {'target': np.ndarray, 'a': np.ndarray}

    def _pour(self, a):
      return {'target': np_func(a), 'a': ut.maybe_copy(a)}

    def _pump(self, target, a):
      return {'a': ut.maybe_copy(a)}

  TankClass.__name__ = class_name

  def func(a, type_dict=None, waterwork=None, name=None):
    type_dict = ut.infer_types(type_dict, a=a)
    return TankClass(a=a, waterwork=waterwork, name=name)

  return func


def create_two_arg_bool_tank(np_func, class_name):
  """Create a function which generates the tank instance corresponding to some two argument, boolean valued numpy function. (e.g. np.equals). The operation will be reversible but in the most trivial and wasteful manner possible. It will just copy over the original array.

  Parameters
  ----------
  np_func : numpy function
      A numpy function which operates on an array to give another array.
  class_name : str
      The name you'd like to have the Tank class called.

  Returns
  -------
  func
      A function which outputs a tank instance which behaves like the np_func but is also reversible

  """

  class TankClass(ta.Tank):
    slot_keys = ['a', 'b']
    tube_dict = {'target': np.ndarray, 'a': np.ndarray, 'b': np.ndarray}

    def _pour(self, a, b):
      return {'target': np_func(a, b), 'a': ut.maybe_copy(a), 'b': ut.maybe_copy(b)}

    def _pump(self, target, a, b):
      return {'a': ut.maybe_copy(a), 'b': ut.maybe_copy(b)}

  TankClass.__name__ = class_name

  def func(a, b, type_dict=None, waterwork=None, name=None):
    type_dict = ut.infer_types(type_dict, a=a)
    return TankClass(a=a, b=b, waterwork=waterwork, name=name)

  return func


isnan = create_one_arg_bool_tank(np.isnan, class_name='IsNan')

equal = create_two_arg_bool_tank(np.equal, class_name='Equals')
greater = create_two_arg_bool_tank(np.greater, class_name='Greater')
greater_equal = create_two_arg_bool_tank(np.greater_equal, class_name='GreaterEqual')
less = create_two_arg_bool_tank(np.less, class_name='Less')
less_equal = create_two_arg_bool_tank(np.less_equal, class_name='LessEqual')
