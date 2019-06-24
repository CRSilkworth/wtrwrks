"""Slot definition."""
import wtrwrks.waterworks.waterwork_part as wp
from wtrwrks.waterworks.empty import empty, Empty
import os

class Slot(wp.WaterworkPart):
  """Object that is always part of some tank which stores the input (in the pour or forward direction) of the operation perfomed by the tank and connects to the tube of another tank.

  Attributes
  ----------
  tank : Tank
    The tank that this slot is a part of.
  key : str
    The string to identify the slot within the tank. Must be unique among all other slots of this tank.
  val : some data type or None
    The value last data inputted to the tank (i.e. operation), if applicable.
  tube : Tube or empty
    The tube from the other tank this tube is connected to, if applicable.
  name : str
    The string used to identify the slot within the entire waterwork. Must be unique among all other slots of this waterwork.
  """
  def __init__(self, tank, key, val=None, tube=empty, plug=None, upstream_slot=None):
    """Initialize the slot.
    Attributes
    ----------
    tank : Tank
      The tank that this slot is a part of.
    key : str
      The string to identify the slot within the tank. Must be unique among all other slots of this tank.
    val : some data type or None
      The value last inputted to the tank (i.e. operation), if applicable.
    tube : Tube or None
      The tube from the other tank this tube is connected to, if applicable.
    """
    self.key = key
    self.tank = tank
    self.tube = tube
    self.name = None
    self.val = val
    self.plug = None
    self.upstream_slot = None

    super(Slot, self).__init__(tank.waterwork, self.name)
    if self.name in self.waterwork.slots:
      raise ValueError(self.name + " already defined as slot. Choose a different name.")
    if plug is not None:
      self.set_plug(plug)
  def __hash__(self):
    """Determine whether two slots are the same within one waterwork."""
    return hash((self.tank, self.key))

  def __eq__(self, other):
    """Determine whether two slots are the same within one waterwork."""
    return (self.tank, self.key) == (other.tank, other.key)

  def __str__(self):
    """Get a string of the name of the slot."""
    return self.name
  #   return str((str(self.tank), str(self.key)))

  def _get_default_name(self, prefix=''):
    """Set a default name. Must be defined by subclass."""
    return os.path.join(self.tank.name, 'slots', self.key)

  def _save_dict(self):
    save_dict = {}
    save_dict['key'] = self.key
    save_dict['tube'] = None if type(self.tube) is Empty else self.tube.name
    save_dict['name'] = self.name
    save_dict['tank'] = self.tank.name
    save_dict['plug'] = self.plug

    return save_dict

  def get_tuple(self):
    """Get a tuple that describes the slot."""
    return (self.tank.name, self.key)

  def get_val(self):
    """Get the value stored in the slot."""
    return self.val

  def set_val(self, val):
    """Set the value stored in the slot."""
    self.val = val

  def set_plug(self, plug, obj_is_callable=False):
    """Plug up the funnel with a function."""

    if self.name not in self.waterwork.funnels:
      raise ValueError("Can only plug funnels. " + str(self) + " is not a funnel.")

    if not callable(plug) or obj_is_callable:
      func_plug = lambda a: plug
    else:
      func_plug = plug

    self.plug = func_plug

  def set_name(self, name):
    """Set the name of the slot within the waterwork."""
    old_name = self.name
    self.name = name

    full_name_space = self.name_space._get_name_string()
    if full_name_space:
      full_name_space = full_name_space + '/'

    if type(name) not in (str, unicode):
      raise TypeError("'name' must be of type str or unicode. Got " + str(type(name)))
    elif not self.name.startswith(full_name_space):
      self.name = os.path.join(self.name_space._get_name_string(), self.name)

    if self.name in self.waterwork.slots:
      raise ValueError(self.name + " already defined as slot. Choose a different name.")

    del self.waterwork.slots[old_name]
    self.waterwork.slots[self.name] = self

    if old_name in self.waterwork.funnels:
      del self.waterwork.funnels[old_name]
      self.waterwork.funnels[self.name] = self

  def unplug(self):
    """Remove the plug function from a funnel."""

    self.plug = None
