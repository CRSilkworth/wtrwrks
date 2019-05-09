import numpy as np
import reversible_transforms.tanks.utils as ut
import reversible_transforms.waterworks.tank as ta


def create_one_arg_reduce_tank(np_func, class_name):
  """Create a function which generates the tank instance corresponding to some single argument, reduceean valued numpy function. (e.g. np.isnan). The operation will be reversible but in the most trivial and wasteful manner possible. It will just copy over the original array.

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

    slot_keys = ['a', 'axis']
    tube_keys = ['target', 'axis', 'a']

    def _pour(self, a, axis):
      if not np.array(axis).size:
        input_axis = None
      else:
        input_axis = axis
      axis = np.array(axis)

      target = np_func(a, axis=input_axis)
      return {'target': target, 'a': ut.maybe_copy(a), 'axis': axis}

    def _pump(self, target, axis, a):
      return {'a': ut.maybe_copy(a), 'axis': axis}

  TankClass.__name__ = class_name

  def func(a, axis=(), type_dict=None, waterwork=None, name=None, return_tank=False):
    tank = TankClass(a=a, axis=axis, waterwork=waterwork, name=name)
    # return tank['target'], tank['axis'], tank['a'], tank.get_slots()
    if not return_tank:
      return tank.get_tubes(), tank.get_slots()
    return tank.get_tubes(), tank.get_slots(), tank
  return func
