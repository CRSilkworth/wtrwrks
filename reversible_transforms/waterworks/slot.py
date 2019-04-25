import reversible_transforms.waterworks.waterwork_part as wp

class Slot(wp.WaterworkPart):
  """Object that is always part of some tank which stores the input (in the pour or forward direction) of the operation perfomed by the tank and connects to the tube of another tank.

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
  name : str
    The string used to identify the slot within the entire waterwork. Must be unique among all other slots of this waterwork.
  """
  def __init__(self, tank, key,  val=None, tube=None):
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
    self.name = str((tank.name, key))
    self.val = val

    super(Slot, self).__init__(tank.waterwork, self.name)
    if self.name in self.waterwork.slots:
      raise ValueError(self.name + " already defined as slot. Choose a different name.")

  def __hash__(self):
    """Determine whether two slots are the same within one waterwork."""
    return hash((self.tank, self.key))

  def __eq__(self, other):
    """Determine whether two slots are the same within one waterwork."""
    return (self.tank, self.key) == (other.tank, other.key)

  def __str__(self):
    """Get a string of the name of the slot."""
    return str((str(self.tank), str(self.key)))

  def get_val(self):
    """Get the value stored in the slot."""
    return self.val

  def set_val(self, val):
    """Set the value stored in the slot."""
    self.val = val
