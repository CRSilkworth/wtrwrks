"""Clone tank definition."""
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import numpy as np
from wtrwrks.waterworks.empty import empty


class DoNothing(ta.Tank):
  """Dummy tank only used if you want to make something part of the waterwork but don't want to actually do anything to it.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """
  func_name = 'do_nothing'
  slot_keys = ['a']
  tube_keys = ['target']

  def _pour(self, a):
    """Execute the DoNothing tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: object
      The object to be added to the waterwork.

    Returns
    -------
    dict(
      target: object
        The object to be added to the waterwork.
    )

    """
    return {'target': a}

  def _pump(self, target):
    """Execute the Clone tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: object
      The object to be added to the waterwork.

    Returns
    -------
    dict(
      a: object
        The object to be added to the waterwork.
    )

    """
    return {'a': target}


class Clone(ta.Tank):
  """The defintion of the Clone tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """
  func_name = 'clone'
  slot_keys = ['a']
  tube_keys = ['a', 'b']

  def _pour(self, a):
    """Execute the Clone tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: object
      The object to be cloned into two.

    Returns
    -------
    dict(
      a: type of slot 'a'
        The first of the two cloned objects.
      b: type of slot 'a'
        The second of the two cloned objects.
    )

    """
    return {'a': ut.maybe_copy(a), 'b': ut.maybe_copy(a)}

  def _pump(self, a, b):
    """Execute the Clone tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    a: type of slot 'a'
      The first of the two cloned objects.
    b: type of slot 'a'
      The second of the two cloned objects.

    Returns
    -------
    dict(
      a: object
        The object to be cloned into two.
    )

    """
    return {'a': ut.maybe_copy(a)}


class CloneMany(ta.Tank):
  """Clone an object many times (rather than just once).

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """
  func_name = 'merge_equal'
  slot_keys = ['a', 'num']
  tube_keys = None
  test_equal = None

  def _pour(self, a):
    """Execute the Clone tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: object
      The object to be cloned into two.
    num: int > 0
      The number of clones

    Returns
    -------
    dict(
      a0: object
        zeroth clone
      a1: object
        first clone,
      .
      .
      .
    )

    """

    r_dict = {}
    for key in self.tube_keys:
      r_dict[key] = ut.maybe_copy(a)
    return r_dict

  def _pump(self, **kwargs):
    """Execute the Clone tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    a0: object
      zeroth clone
    a1: object
      first clone,
    .
    .
    .

    Returns
    -------
    dict(
      a: object
        The object to be cloned into two.
    )

    """
    r_dict = {}
    for key in self.slot_keys:
      r_dict['a'] = ut.maybe_copy(kwargs[key])
      break
    return r_dict


class MergeEqual(ta.Tank):
  """Merge several equal objects into a single object. All tubes must have equal values, otherwise you will get unexpected results. (Opposite of clone_many)

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """
  func_name = 'merge_equal'
  slot_keys = None
  tube_keys = ['target']
  test_equal = None

  def _pour(self, **kwargs):
    """Execute the MergeEqual tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a0: object
      zeroth equal object
    a1: object
      first equal object,
    .
    .
    .

    Returns
    -------
    dict(
      a: type of slot 'a'target: object
        The merged object. Simply takes the value of the first in the list.
    )

    """
    if self.test_equal:
      for key in kwargs:
        if not np.all(kwargs[key] == kwargs['a0']):
          raise ValueError("All arguments passed to merge_equal must be equal. Got " + str(kwargs[key]) + ' and ' + str(kwargs['a0']))
    return {'target': ut.maybe_copy(kwargs['a0'])}

  def _pump(self, target):
    """Execute the Clone tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: object
      The merged object. Simply takes the value of the first in the list.

    Returns
    -------
    dict(
      a0: object
        zeroth equal object
      a1: object
        first equal object,
      .
      .
      .
    )

    """
    kwargs = {}
    for key in self.slot_keys:
      kwargs[key] = ut.maybe_copy(target)
    return kwargs

  def _save_dict(self):
    save_dict = {}
    save_dict['func_name'] = self.func_name
    save_dict['name'] = self.name
    save_dict['args'] = [empty for _ in self.slot_keys]

    return save_dict
