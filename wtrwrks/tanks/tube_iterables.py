"""IterList and IterDict tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import numpy as np


class TubeList(ta.Tank):
  """The defintion of the IterList tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = '_tube_list'
  slot_keys = None
  tube_keys = ['target']

  def _pour(self, **kwargs):
    """Execute the IterList tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a1: object
      first element of the list
    .
    .
    .
    an: object
      nth element of the list

    Returns
    -------
    dict(
      a: list
        list of a1...an
    )

    """
    r_list = []
    for key in self.slot_keys:
      r_list.append(kwargs[key])
    return {'target': r_list}

  def _pump(self, target):
    """Execute the IterList tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    a: list
      a: list of a1...an

    Returns
    -------
    dict(
      a1: object
        first element of the list
      .
      .
      .
      an: object
        nth element of the list
    )

    """
    r_dict = {}

    for num, slot_key in enumerate(self.slot_keys):
      r_dict[slot_key] = target[num]

    return r_dict

  def _save_dict(self):
    save_dict = {}
    save_dict['func_name'] = self.func_name
    save_dict['name'] = self.name
    # save_dict['slots'] = [s for s in self.get_slots()]
    # save_dict['tubes'] = [t for t in self.get_tubes()]
    save_dict['kwargs'] = {s: None for s in self.slot_keys}
    return save_dict
