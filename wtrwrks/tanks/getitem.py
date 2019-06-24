"""GetItem tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut


class GetItem(ta.Tank):
  """Run __getitem__ on some object (e.g. a dictionary) to return some value.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'getitem'
  slot_keys = ['a', 'key']
  tube_keys = ['target', 'a', 'key']
  pass_through_keys = ['a', 'key']

  def _pour(self, a, key):
    """

    Parameters
    ----------
    a: object
      The object to getitem from.
    key: hashable
      The key to pass to the getitem

    Returns
    -------
    dict(
      target: object
        The value returned from the __getitem__ call to 'a'.
      a: object
        The object to getitem from.
      key: hashable
        The key to pass to the getitem
    )

    """
    return {'target': a[key], 'a': ut.maybe_copy(a)}

  def _pump(self, target, a, key):
    """Execute the Shape tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: object
      The value returned from the __getitem__ call to 'a'.
    a: object
      The object to getitem from.
    key: hashable
      The key to pass to the getitem

    Returns
    -------
    dict(
      a: object
        The object to getitem from.
      key: hashable
        The key to pass to the getitem
    )

    """
    return {'a': ut.maybe_copy(a), 'key': key}

  # def _save_dict(self):
  #   save_dict = {}
  #   save_dict['func_name'] = self.func_name
  #   save_dict['name'] = self.name
  #   # save_dict['slots'] = [s for s in self.get_slots()]
  #   # save_dict['tubes'] = [t for t in self.get_tubes()]
  #   save_dict['kwargs'] = {'num_entries': len(self.tube_keys)}
  #   return save_dict
