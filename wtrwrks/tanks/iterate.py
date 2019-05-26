"""IterList and IterDict tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import numpy as np


class IterList(ta.Tank):
  """The defintion of the IterList tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  slot_keys = ['a']
  tube_keys = None

  def _pour(self, a):
    """Execute the IterList tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: list
      The tube list whose elements will be converted to tubes.

    Returns
    -------
    dict(
      a: list of values
    )

    """
    r_dict = {}
    for key_num, tube_key in enumerate(self.tube_keys):
      r_dict[tube_key] = a[key_num]
    return r_dict

  def _pump(self, **kwargs):
    """Execute the IterList tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    kwargs: list of values

    Returns
    -------
    dict(
      a: list
        The tube list whose elements will be converted to tubes.
    )

    """
    a = []
    for tube_key in self.tube_keys:
      a.append(kwargs[tube_key])
    return {'a': a}


class IterDict(ta.Tank):
  """The defintion of the IterDict tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  slot_keys = ['a']
  tube_keys = None

  def _pour(self, a):
    """Execute the IterDict tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: dict
      The tube dictionary. whose values will be converted to tubes.

    Returns
    -------
    dict(
      a: dict
        The dictionary of values
    )

    """
    r_dict = {}
    for tube_key in self.tube_keys:
      r_dict[tube_key] = a[tube_key]
    return r_dict

  def _pump(self, **kwargs):
    """Execute the IterDict tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    kwargs: dict
      The dictionary of values

    Returns
    -------
    dict(
      a: dict
        The tube dictionary. whose values will be converted to tubes.
    )

    """
    return {'a': kwargs}
