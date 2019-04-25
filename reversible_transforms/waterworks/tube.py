import reversible_transforms.waterworks.waterwork_part as wp

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
  def __init__(self, tank, key, val=None, slot=None):
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
    self.name = str((tank.name, key))
    self.val = val

    super(Tube, self).__init__(tank.waterwork, self.name)
    if self.name in self.waterwork.tubes:
      raise ValueError(self.name + " already defined as tube. Choose a different name.")

  def __eq__(self, other):
    """Determine whether two tubes are the same within one waterwork."""
    return (self.tank, self.key) == (other.tank, other.key)

  def __hash__(self):
    """Determine whether two tubes are the same within one waterwork."""
    return hash((self.tank, self.key))

  def __str__(self):
    """Get a string of the name of the tube."""
    return str((str(self.tank), str(self.key)))

  def get_val(self):
    """Get the value stored in the tube."""
    return self.val

  def set_val(self, val):
    """Set the value stored in the tube."""
    self.val = val
