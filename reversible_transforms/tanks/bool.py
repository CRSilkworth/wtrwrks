import numpy as np
import reversible_transforms.tanks.utils as ut
import reversible_transforms.waterworks.tank as ta
from reversible_transforms.waterworks.empty import empty


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
    tube_keys = ['target', 'a']

    def _pour(self, a):
      return {'target': np_func(a), 'a': ut.maybe_copy(a)}

    def _pump(self, target, a):
      return {'a': ut.maybe_copy(a)}

  TankClass.__name__ = class_name

  def func(a=empty, type_dict=None, waterwork=None, name=None, return_tank=False):
    tank = TankClass(a=a, waterwork=waterwork, name=name)
    # return tank['target'], tank['a'], tank.get_slots()
    if not return_tank:
      return tank.get_tubes(), tank.get_slots()
    return tank.get_tubes(), tank.get_slots(), tank
  return func


def create_two_arg_bool_tank(np_func, class_name, target_type=None):
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

  # Define the tank subclass.
  class TankClass(ta.Tank):
    slot_keys = ['a', 'b']
    tube_keys = ['target', 'a', 'b']

    def _pour(self, a, b):
      return {'target': np_func(a, b), 'a': ut.maybe_copy(a), 'b': ut.maybe_copy(b)}

    def _pump(self, target, a, b):
      return {'a': ut.maybe_copy(a), 'b': ut.maybe_copy(b)}

  TankClass.__name__ = class_name

  def func(a=empty, b=empty, type_dict=None, waterwork=None, name=None, return_tank=False):
    tank = TankClass(a=a, b=b, waterwork=waterwork, name=name)
    # return tank['target'], tank['a'], tank['b'], tank.get_slots()
    if not return_tank:
      return tank.get_tubes(), tank.get_slots()
    return tank.get_tubes(), tank.get_slots(), tank
  return func


class LogicalNot(ta.Tank):
  slot_keys = ['a']
  tube_keys = ['target']

  def _pour(self, a):
    a = np.array(a)
    return {'target': np.logical_not(a)}

  def _pump(self, target):
    return {'a': np.logical_not(target)}
