"""Functions that create tanks that operate on or create boolean arrays."""
import numpy as np
import wtrwrks.tanks.utils as ut
import wtrwrks.waterworks.tank as ta
from wtrwrks.waterworks.empty import empty


def create_one_arg_bool_tank(np_func, class_name, func_name):
  """Create a function which generates the tank instance corresponding to some single argument, boolean valued numpy function. (e.g. np.isnan). The operation will be reversible but in the most trivial and wasteful manner possible. It will just copy over the original array.

  Parameters
  ----------
  np_func : numpy function
      A numpy function which operates on an array to give another array.
  class_name : str
      The name you'd like to have the Tank class called.
  func_name : str
    The name of function which actually creates the Tank. (e.g. the functions found in tank_def.py)

  Returns
  -------
  func
      A function which outputs a tank instance which behaves like the np_func but is also reversible

  """
  temp = func_name

  # Define the tank subclass.
  class TankClass(ta.Tank):
    func_name = temp
    slot_keys = ['a']
    tube_keys = ['target', 'a']
    pass_through_keys = ['a']

    def _pour(self, a):
      return {'target': np_func(a), 'a': ut.maybe_copy(a)}

    def _pump(self, target, a):
      return {'a': ut.maybe_copy(a)}

  TankClass.__name__ = class_name

  def func(a=empty, type_dict=None, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
    tank = TankClass(a=a, waterwork=waterwork, name=name)

    if slot_plugs is not None:
      for key in slot_plugs:
        tank.get_slots()[key].set_plug(slot_plugs[key])
    if tube_plugs is not None:
      for key in tube_plugs:
        tank.get_tubes()[key].set_plug(tube_plugs[key])
    if slot_names is not None:
      for key in slot_names:
        tank.get_slots()[key].set_name(slot_names[key])
    if tube_names is not None:
      for key in tube_names:
        tank.get_tubes()[key].set_name(tube_names[key])
    return tank.get_tubes(), tank.get_slots()
  return func


def create_two_arg_bool_tank(np_func, class_name, func_name):
  """Create a function which generates the tank instance corresponding to some two argument, boolean valued numpy function. (e.g. np.equals). The operation will be reversible but in the most trivial and wasteful manner possible. It will just copy over the original array.

  Parameters
  ----------
  np_func : numpy function
      A numpy function which operates on an array to give another array.
  class_name : str
      The name you'd like to have the Tank class called.
  func_name : str
    The name of function which actually creates the Tank. (e.g. the functions found in tank_def.py)
  Returns
  -------
  func
      A function which outputs a tank instance which behaves like the np_func but is also reversible

  """
  temp = func_name

  # Define the tank subclass.
  class TankClass(ta.Tank):
    func_name = temp
    slot_keys = ['a', 'b']
    tube_keys = ['target', 'a', 'b']
    pass_through_keys = ['a', 'b']

    def _pour(self, a, b):
      return {'target': np_func(a, b), 'a': ut.maybe_copy(a), 'b': ut.maybe_copy(b)}

    def _pump(self, target, a, b):
      return {'a': ut.maybe_copy(a), 'b': ut.maybe_copy(b)}

  TankClass.__name__ = class_name

  def func(a=empty, b=empty, type_dict=None, waterwork=None, name=None, slot_plugs=None, tube_plugs=None, slot_names=None, tube_names=None):
    tank = TankClass(a=a, b=b, waterwork=waterwork, name=name)
    if slot_plugs is not None:
      for key in slot_plugs:
        tank.get_slots()[key].set_plug(slot_plugs[key])
    if tube_plugs is not None:
      for key in tube_plugs:
        tank.get_tubes()[key].set_plug(tube_plugs[key])
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
  func_name = 'logical_not'
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


class Equals(ta.Tank):
  """The defintion of the LogicalNot tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """
  func_name = 'equals'
  slot_keys = ['a', 'b']
  tube_keys = ['target', 'a', 'b']
  pass_through_keys = ['a', 'b']
  def _pour(self, a, b):
    """Test the equality of a and b.

    Parameters
    ----------
    a: object
      The first object in the equal operation
    b: object
      The second object in the equal operation

    Returns
    -------
    dict(
      target: object.
        The results of a == b
      a: object
        The first object in the equal operation
      b: object
        The second object in the equal operation
    )

    """
    a = np.array(a, copy=True)
    b = np.array(b, copy=True)

    return {'target': a == b, 'a': a, 'b': b}

  def _pump(self, target, a, b):
    """Execute the LogicalNot tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: object.
      The results of a == b
    a: object
      The first object in the equal operation
    b: object
      The second object in the equal operation

    Returns
    -------
    dict(
      a: object
        The first object in the equal operation
      b: object
        The second object in the equal operation
    )

    """
    a = np.array(a, copy=True)
    b = np.array(b, copy=True)
    return {'a': a, 'b': b}


class MultiIsIn(ta.Tank):
  func_name = 'multi_isin'
  slot_keys = ['a', 'bs', 'selector']
  tube_keys = ['target', 'a', 'bs', 'selector']
  pass_through_keys = ['a', 'bs', 'selector']

  def _pour(self, a, bs, selector):
    if a.shape != selector.shape:
      raise ValueError("Shape of a and selector must match. Got {} and {}".format(a.shape, selector.shape))

    uniques = np.unique(selector)
    target = np.zeros(a.shape, dtype=bool)
    for unique in uniques:
      mask = selector == unique
      target[mask] = np.isin(a[mask], bs[unique])

    return {'target': target, 'a': ut.maybe_copy(a), 'bs': ut.maybe_copy(bs), 'selector': ut.maybe_copy(selector)}

  def _pump(self, target, a, bs, selector):
    return {'a': ut.maybe_copy(a), 'bs': ut.maybe_copy(bs), 'selector': ut.maybe_copy(selector)}
