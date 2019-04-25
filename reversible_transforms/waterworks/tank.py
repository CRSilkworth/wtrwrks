import reversible_transforms.waterworks.globs as gl
import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.slot as sl
import reversible_transforms.waterworks.tube as tu
import os
import pprint


class Tank(wp.WaterworkPart):
  """Base class for any tank, i.e. an operation that pulls in information from the 'slots', does some processing and outputs them to the 'tubes'.

  Attributes
  ----------
  waterwork : Waterwork or None
    The waterwork that the part will be added to. If None, it is assinged to the _default_waterwork.
  name : str or None
    The name of the part within the waterwork. Must be unique. If None, it will be set to a default value depending on the subclass.
  slot_keys : list of str
    The tank's (operation's) argument keys. They define the names of the inputs to the tank.
  tube_keys : list of str
    The tank's (operation's) output keys. They define the names of tanks outputs.
  slots : dict({
    keys - strs. equal to that of the slot_keys.
    values - Slot object.
  })
    The slot objects that define the pour direction inputs (or pump direction outputs) of the tank.
  tubes : dict({
    keys - strs. equal to that of the tube_keys.
    values - Slot object.
  })
    The tube objects that define the pour direction outputs (or pump direction inputs) of the tank.
  """

  slot_keys = None
  tube_keys = None

  def __init__(self, waterwork=None, name=None, **input_dict):
    """Create a Tank. Eagerly run the pour function if all the input values are known at creation.

    Parameters
    ----------
    waterwork : type
        Description of parameter `waterwork`.

    name : type
        Description of parameter `name`.

    **input_dict : kwargs = {
        keys - Slot keys. Must be the same as the attribute slot_keys.
        values - Tube, None or some valid input data type.
      }
      The inputs to the tank. If the value is of Tube type then the slot referenced by the key is 'connected' to that tube. If the value is None then it will be left empty and will become a 'funnel' (i.e. a slot that needs inputted data to run.) when running the Waterwork the tank belongs to. If all the values are either valid input data types or tubes with the values filled, then it will eagerly run the tank and fill all the relevant tubes.

    """
    # Use the WaterworkPart constructor to assign the waterwork and the name.
    super(Tank, self).__init__(waterwork, name)
    if self.name in self.waterwork.tanks:
      raise ValueError(self.name + " already defined as tank. Choose a different name.")

    # Assign self the the waterwork's directory of tanks.
    self.waterwork.tanks[self.name] = self

    # Make sure all the inputed arguments are valid.
    for key in input_dict:
      if key not in self.slot_keys:
        raise TypeError(key + ' not a valid argument for ' + str(type(self)))

    # Create all the slots and tubes for this tank.
    self.slots = {}
    self.tubes = {}
    self._create_slots(self.slot_keys, self.waterwork)
    self._create_tubes(self.tube_keys, self.waterwork)

    # Join the this tank's slots to the tubes of the other tanks which are
    # inputted in as an argument (input_dict).
    self._join_tubes_to_slots(input_dict, self.waterwork)

    # If all the slots of the tank are 'filled', i.e. are either connected to a
    # tube with a non None val or are given a valid datum as input, then
    # eagerly run the tank's pour function and output the results to the tank's
    # tubes' vals.
    all_slots_filled = self._check_slots_filled(input_dict)
    if all_slots_filled:
      input_dict = self._convert_tubes_to_vals(input_dict)
      self.pour(**input_dict)
      for key in input_dict:
        self.slots[key].set_val(input_dict[key])

  def __hash__(self):
    """Uniquely identify the tank among other tanks in the waterwork."""
    return hash(self.name)

  def __getitem__(self, key):
    """Return a tube object of the tank, denonted by the key."""
    return self.tubes[key]

  def __str__(self):
    """Return the str of the tank, which is just its name."""
    return str(self.name)

  def _check_slots_filled(self, input_dict):
    """Check whether or not the all the values of the input_dict are set to determine whether or not the tank can be eagerly executed.

    Parameters
    ----------
    input_dict : dict({
        keys - Slot keys. Must be the same as the attribute slot_keys.
        values - Tube, None or some valid input data type.
      })
      The inputs to the tank.

    Returns
    -------
    bool
        Whether or not all the values needed for eager execution have been filled.

    """

    # If there are any slot keys that are not covered by the input_dict, then
    # they have not all been filled.
    if set(self.slot_keys) - set(input_dict.keys()):
      return False

    # Go through each value of the input dict. If there are any Nones, or there
    # are any Slot objects that don't have a value, then return False.
    # Otherwise return True.
    all_slots_filled = True
    for key in input_dict:
      if input_dict[key] is None:
        all_slots_filled = False
        break
      if (
        type(input_dict[key]) is tu.Tube and
        input_dict[key].get_val() is None):
        all_slots_filled = False
        break
    return all_slots_filled

  def _convert_tubes_to_vals(self, input_dict):
    """Pull out the values associated with the tubes connected to this tank's slots. Where the values of the dictionary are the values stored in the tube, rather than the Tube object itself.

    Parameters
    ----------
    input_dict : dict({
        keys - Slot keys. Must be the same as the attribute slot_keys.
        values - Tube, None or some valid input data type.
      })
      The inputs to the tank.

    Returns
    -------
    dict({
      keys - Slot keys. Must be the same as the attribute slot_keys.
      values - valid input data type.
    })
        The inputted dict with the Tube objects replaced by the 'val' attribute of the tube.

    """
    r_dict = {}
    r_dict.update(input_dict)
    for key in input_dict:
      if type(input_dict[key]) is tu.Tube:
        r_dict[key] = input_dict[key].get_val()

    return r_dict

  def _create_slots(self, slot_keys, waterwork):
    """Create all the slot objects for the tank and add to the waterwork directory.

    Parameters
    ----------
    slot_keys : list of str
      The tank's (operation's) argument keys. They define the names of the inputs to the tank.

    waterwork : Waterwork
      The waterwork that the part will be added to.

    """
    for key in slot_keys:
      slot = sl.Slot(self, key)
      self.slots[key] = slot

      waterwork.slots[slot.name] = slot
      waterwork.funnels[slot.name] = slot

  def _create_tubes(self, tube_keys, waterwork):
    """Create all the tube objects for the tank and add to the waterwork directory.

    Parameters
    ----------
    slot_keys : list of str
      The tank's (operation's) argument keys. They define the names of the inputs to the tank.

    waterwork : Waterwork
      The waterwork that the part will be added to.

    """
    for key in tube_keys:
      tube = tu.Tube(self, key)
      self.tubes[key] = tube

      waterwork.tubes[tube.name] = tube
      waterwork.taps[tube.name] = tube

  def _get_default_name(self, prefix=''):
    """Create the default name of the tank, of the form '<TankSubClass>_<num>'.


    Returns
    -------
    str
        The name of the tank.

    """
    num = 0
    cls_name = self.__class__.__name__

    # Start with the name being '<TankSubClass>_0'. If that is already taken,
    # keep increasing the number from 0 until an unused name is found.
    full_name = os.path.join(prefix, cls_name + '_' + str(num))
    while full_name in gl._default_waterwork.tanks:
      num += 1
      full_name = os.path.join(prefix, cls_name + '_' + str(num))

    return full_name

  def _join_tubes_to_slots(self, input_dict, waterwork):
    """Join the tubes incoming from other tanks to this tank's slots. If the slot was previously identified as a 'funnel', i.e. a slot not attached to other tube then take it out of the list of funnels. If the tube was previously identified as a tap, i.e. a tube that is not connected to another slot, then take it out of the list of taps.

    Parameters
    ----------
    input_dict : dict({
        keys - Slot keys. Must be the same as the attribute slot_keys.
        values - Tube, None or some valid input data type.
      })
      The inputs to the tank.
    waterwork : Waterwork
      The waterwork that the part will be added to.

    Returns
    -------
    type
        Description of returned object.

    """
    for key in input_dict:
      slot = self.slots[key]
      tube = input_dict[key]

      if type(tube) is not tu.Tube:
        continue

      if tube.slot is not None:
        other_slot = tube.slot
        clone = Clone(a=tube)

        other_slot.tube = clone['a']
        slot.tube = clone['a']

      tube.slot = slot
      slot.tube = tube

      del waterwork.funnels[slot.name]
      del waterwork.taps[tube.name]

  def _pour(self, *args, **kwargs):
    """Make sure pour function is set by subclass. The forward transformation of inputs."""
    raise ValueError("'_pour' method not defined for " + str(type(self)))

  def _pump(self, *args, **kwargs):
    """Make sure pump function is set by subclass. The backward transformation of outputs."""
    raise ValueError("'_pump' method not defined for " + str(type(self)))

  def get_slot(self, key):
    """Retrieve a slot object from the tank identified by the key"""
    return self.slots[key]

  def get_slots(self):
    """Retrieve the dictionary of slot objects from the tank."""
    slots = {}
    slots.update(self.slots)
    return slots

  def get_tube(self, key):
    """Retrieve a tube object from the tank identified by the key"""
    return self.tubes[key]

  def get_tubes(self):
    """Retrieve the dictionary of tube objects from the tank."""
    tubes = {}
    tubes.update(self.tubes)
    return tubes

  def paired_slots(self):
    """Retrieve the dictionary of slot objects from the tank which have been paired to a tube from another tank."""
    slots = {}
    slots.update({k: s for k, s in self.slots.iteritems() if s.tube is not None})
    return slots

  def paired_tubes(self):
    """Retrieve the dictionary of tube objects from the tank which have been paired to a slot from another tank."""
    tubes = {}
    tubes.update({k: t for k, t in self.tubes.iteritems() if t.slot is not None})
    return tubes

  def unpaired_slots(self):
    """Retrieve the dictionary of slot objects from the tank which have not been paired to a tube from another tank."""
    slots = {}
    slots.update({k: s for k, s in self.slots.iteritems() if s.tube is None})
    return slots

  def unpaired_tubes(self):
    """Retrieve the dictionary of tube objects from the tank which have not been paired to a slot from another tank."""
    tubes = {}
    tubes.update({k: t for k, t in self.tubes.iteritems() if t.slot is None})
    return tubes

  def pour(self, **input_dict):
    """Execute the forward transformation of the input_dict inputted to the tank to get the dictionary of tube objects who's val's have been filled.

    Parameters
    ----------
    **input_dict : kwargs = {
        keys - Slot keys. Must be the same as the attribute slot_keys.
        values - valid input data types
      }
      The inputs to the tank.

    Returns
    -------
    kwargs = {
        keys - Tube keys. The same as the attribute tube_keys.
        values - The data_types outputted by the tank.
      }
        All of the ouputs the tank gives in the 'pour' (i.e. forward) direction.

    """
    # Check that the inputs are valid
    if set(input_dict.keys()) != set(self.slot_keys):
      raise ValueError("Must pass " + str(input_dict.keys()) + " as arguments, got " + str(self.slot_keys))

    # Run the function defined by the subclass
    tube_dict = self._pour(**input_dict)

    # Set the vals
    for key in tube_dict:
      self.tubes[key].set_val(tube_dict[key])

    return tube_dict

  def pump(self, **kwargs):
    """Execute the backward transformation of the kwargs inputted to the tank to get the dictionary of slot objects who's val's have been filled.

    Parameters
    ----------
    **kwargs : kwargs = {
        keys - Tube keys. Must be the same as the attribute tube_keys.
        values - valid data types
      }
      The inputs to the backward transformation of the tank.

    Returns
    -------
    kwargs = {
        keys - Slot keys. The same as the attribute slot_keys.
        values - The data_types outputted by the tank.
      }
        All of the ouputs the tank gives in the 'pump' (i.e. backward) direction.

    """
    # Check that the inputs are valid
    if set(kwargs.keys()) != set(self.tube_keys):
      raise ValueError("Must pass " + str(kwargs.keys()) + " as arguments, got " + str(self.tube_keys))

    # Run the function defined by the subclass
    slot_dict = self._pump(**kwargs)

    # Set the vals
    for key in slot_dict:
      self.slots[key].set_val(slot_dict[key])


class Clone(Tank):
  slot_keys = ['a']
  tube_keys = ['a', 'b']

  def _pour(self, a):
    return {'a': a, 'b': a}

  def _pump(self, a, b):
    return {'a': a}
