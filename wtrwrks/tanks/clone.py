"""Clone tank definition."""
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut


class Clone(ta.Tank):
  """The defintion of the Clone tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """
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
