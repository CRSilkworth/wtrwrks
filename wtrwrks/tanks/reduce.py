import numpy as np
import wtrwrks.tanks.utils as ut
import wtrwrks.waterworks.tank as ta


def create_one_arg_reduce_tank(np_func, class_name, func_name):
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
  temp = func_name

  # Define the tank subclass.
  class TankClass(ta.Tank):
    func_name = temp
    slot_keys = ['a', 'axis']
    tube_keys = ['target', 'axis', 'a']
    pass_through_keys = ['a', 'axis']

    def _pour(self, a, axis):
      # If an empty tuple was given then set the axis to None
      if not np.array(axis).size:
        input_axis = None
      else:
        input_axis = axis
      axis = np.array(axis)

      # Reduce the array using the supplied numpy array function.
      target = np_func(a, axis=input_axis)
      return {'target': target, 'a': ut.maybe_copy(a), 'axis': axis}

    def _pump(self, target, axis, a):
      return {'a': ut.maybe_copy(a), 'axis': axis}

  # Set the name of the tank class
  TankClass.__name__ = class_name

  # Return the tank_def function.
  def func(a, axis=(), type_dict=None, waterwork=None, name=None, return_tank=False, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
    tank = TankClass(a=a, axis=axis, waterwork=waterwork, name=name)
    if slot_plugs is not None:
      for key in slot_plugs:
        tank.get_slots()[key].set_plug(slot_plugs[key])
    if tube_plugs is not None:
      for key in tube_plugs:
        tank.get_tubes()[key].set_plug(tube_plugs[key])
    return tank.get_tubes(), tank.get_slots()
  return func
