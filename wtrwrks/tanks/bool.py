"""Functions that create tanks that operate on or create boolean arrays."""
import numpy as np
import wtrwrks.tanks.utils as ut
import wtrwrks.waterworks.tank as ta
from wtrwrks.waterworks.empty import empty


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

  def func(a=empty, type_dict=None, waterwork=None, name=None):
    tank = TankClass(a=a, waterwork=waterwork, name=name)
    return tank.get_tubes(), tank.get_slots()
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

  def func(a=empty, b=empty, type_dict=None, waterwork=None, name=None):
    tank = TankClass(a=a, b=b, waterwork=waterwork, name=name)
    return tank.get_tubes(), tank.get_slots()
  return func


class LogicalNot(ta.Tank):
  """The defintion of the LogicalNot tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """
  slot_keys = ['a']
  tube_keys = ['target']

  def _pour(self, a):
    """Execute the LogicalNot tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: np.ndarray of bools
      The array to take the logical not of.

    Returns
    -------
    dict(
      target: np.ndarray of bools.
        The negated array.
    )

    """
    a = np.array(a)
    return {'target': np.logical_not(a)}

  def _pump(self, target):
    """Execute the LogicalNot tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray of bools.
      The negated array.

    Returns
    -------
    dict(
      a: np.ndarray of bools
        The array to take the logical not of.
    )

    """
    return {'a': np.logical_not(target)}
