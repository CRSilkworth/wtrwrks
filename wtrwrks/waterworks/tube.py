"""Tube definition."""
import wtrwrks.waterworks.waterwork_part as wp
from wtrwrks.waterworks.empty import Empty, empty
import os

class Tube(wp.WaterworkPart):
  """Object that is always part of some tank which stores the output (in the pour or forward direction) of the operation perfomed by the tank and connects to the slot of another tank.

  Attributes
  ----------
  tank : Tank
    The tank that this tube is a part of.
  key : str
    The string to identify the tube within the tank. Must be unique among all other tubes of this tank.
  val : some data type or None
    The value last outputed by the tank (i.e. operation), if applicable.
  slot : Slot or None
    The slot from the other tank this tube is connected to, if applicable.
  name : str
    The string used to identify the tube within the entire waterwork. Must be unique among all other tubes of this waterwork.

  """

  def __init__(self, tank, key, val=None, slot=empty, plug=None, downstream_tube=None):
    """Initialize the tube.

    Parameters
    ----------
    tank : Tank
      The tank that this tube is a part of.
    key : str
      The string to identify the tube within the tank. Must be unique among all other tubes of this tank.
    val : some data type or None
      The value last outputed by the tank (i.e. operation), if applicable.
    slot : Slot or None
      The slot from the other tank this tube is connected to, if applicable.
    """
    self.key = key
    self.tank = tank
    self.slot = slot
    self.name = None
    self.val = val
    self.plug = None
    self.downstream_tube = downstream_tube

    super(Tube, self).__init__(tank.waterwork, self.name)
    if self.name in self.waterwork.tubes:
      raise ValueError(self.name + " already defined as tube. Choose a different name.")
    if plug is not None:
      self.set_plug(plug)

  def __add__(self, other):
    """Define an add tank (operation) between two tubes."""
    import wtrwrks.tanks.tank_defs as td
    return td.add(a=self, b=other, waterwork=self.waterwork)

  def __div__(self, other):
    """Define an add tank (operation) between two tubes."""
    import wtrwrks.tanks.tank_defs as td
    return td.div(a=self, b=other, waterwork=self.waterwork)

  def __eq__(self, other):
    """Determine whether two tubes are the same within one waterwork."""
    return (self.tank, self.key) == (other.tank, other.key)

  def __hash__(self):
    """Determine whether two tubes are the same within one waterwork."""
    return hash((self.tank, self.key))

  def __mul__(self, other):
    """Define an add tank (operation) between two tubes."""
    import wtrwrks.tanks.tank_defs as td
    return td.mul(a=self, b=other, waterwork=self.waterwork)

  def __radd__(self, other):
    """Define an add tank (operation) between two tubes."""
    import wtrwrks.tanks.tank_defs as td
    return td.add(a=other, b=self, waterwork=self.waterwork)

  def __rdiv__(self, other):
    """Define an add tank (operation) between two tubes."""
    import wtrwrks.tanks.tank_defs as td
    return td.div(a=other, b=self, waterwork=self.waterwork)

  def __rmul__(self, other):
    """Define an add tank (operation) between two tubes."""
    import wtrwrks.tanks.tank_defs as td
    return td.mul(a=other, b=self, waterwork=self.waterwork)

  def __rsub__(self, other):
    """Define an add tank (operation) between two tubes."""
    import wtrwrks.tanks.tank_defs as td
    return td.sub(a=other, b=self, waterwork=self.waterwork)

  def __str__(self):
    """Get a string of the name of the tube."""
    return self.name

  def __sub__(self, other):
    """Define an add tank (operation) between two tubes."""
    import wtrwrks.tanks.tank_defs as td
    return td.sub(a=self, b=other, waterwork=self.waterwork)

  def _get_default_name(self, prefix=''):
    """Set a default name. Must be defined by subclass."""
    return os.path.join(self.tank.name, 'tubes', self.key)

  def _save_dict(self):
    save_dict = {}
    save_dict['key'] = self.key
    save_dict['slot'] = None if type(self.slot) is Empty else self.slot.name
    save_dict['name'] = self.name
    save_dict['tank'] = self.tank.name
    save_dict['plug'] = self.plug
    save_dict['downstream_tube'] = None if self.downstream_tube is None else self.downstream_tube.name

    return save_dict

  def get_tuple(self):
    """Get a tuple that describes the tube."""
    return (self.tank.name, self.key)

  def get_val(self):
    """Get the value stored in the tube."""
    return self.val

  def set_val(self, val):
    """Set the value stored in the tube."""
    self.val = val

  def set_plug(self, plug, downstream=True):
    """Plug up the tap with a function."""
    if self.downstream_tube is not None and downstream:
      self = self.downstream_tube

    if self.name not in self.waterwork.taps:
      raise ValueError("Can only plug taps. " + str(self) + " is not a tap.")

    if not callable(plug):
      func_plug = lambda a: plug
    else:
      func_plug = plug
    self.plug = func_plug

  def set_name(self, name, downstream=True):
    """Set the name of the tube within the waterwork."""
    # If it has a downstream tube, then actually set that name. Unless
    # downstream is set to false
    tu = self
    if self.downstream_tube is not None and downstream:
      tu = self.downstream_tube

    old_name = tu.name
    tu.name = name

    full_name_space = tu.name_space._get_name_string()
    if full_name_space:
      full_name_space = full_name_space + '/'

    if type(name) not in (str, unicode):
      raise TypeError("'name' must be of type str or unicode. Got " + str(type(name)))
    elif not tu.name.startswith(full_name_space):
      tu.name = os.path.join(tu.name_space._get_name_string(), tu.name)

    if tu.name in tu.waterwork.tubes:
      raise ValueError(tu.name + " already defined as tube. Choose a different name.")

    del tu.waterwork.tubes[old_name]
    tu.waterwork.tubes[tu.name] = tu

    if old_name in tu.waterwork.taps:
      del tu.waterwork.taps[old_name]
      tu.waterwork.taps[tu.name] = tu

  def unplug(self, downstream=True):
    """Remove the plug function from a tap."""
    if self.downstream_tube is not None and downstream:
      self = self.downstream_tube
    self.plug = None
