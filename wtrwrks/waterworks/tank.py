"""Tank definition."""
import wtrwrks.waterworks.globs as gl
import wtrwrks.waterworks.waterwork_part as wp
from wtrwrks.waterworks.empty import Empty, empty
import wtrwrks.waterworks.slot as sl
import wtrwrks.waterworks.tube as tu
import os
import sys


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
  tube_dict : dict(
    keys - strs. The tank's (operation's) output keys. THey define the names of the outputs of the tank
    values - types. The types of the arguments outputs.
  )
    The tank's (operation's) output keys and their corresponding types.
  slots : dict(
    keys - strs. equal to that of the slot_keys.
    values - Slot object.
  )
    The slot objects that define the pour direction inputs (or pump direction outputs) of the tank.
  tubes : dict(
    keys - strs. equal to that of the tube_keys.
    values - Slot object.
  )
    The tube objects that define the pour direction outputs (or pump direction inputs) of the tank.

  """
  func_name = None
  slot_keys = None
  tube_keys = None
  pass_through_keys = None

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
    for key in input_dict:
      if type(input_dict[key]) is sl.Slot:
        raise ValueError("Cannot pass slot as argument to " + str(type(self)))
      if type(input_dict[key]) is list or type(input_dict[key]) is tuple:
        input_dict[key] = self._handle_iterable(input_dict[key])
      if type(input_dict[key]) is tu.Tube and input_dict[key].downstream_tube is not None:
        input_dict[key] = input_dict[key].downstream_tube

    self._join_tubes_to_slots(input_dict, self.waterwork)

    # If all the slots of the tank are 'filled', i.e. are either connected to a
    # tube with a non None val or are given a valid datum as input, then
    # eagerly run the tank's pour function and output the results to the tank's
    # tubes' vals.

    all_slots_filled = self._check_slots_filled(input_dict)

    for key in input_dict:
      if type(input_dict[key]) is sl.Slot:
        raise ValueError("Cannot pass slot as argument to " + str(type(self)))

      # If data is directly inputted into the slot, then create a placeholder
      # with it's value set to the data. And set it as plugged to that value by # default.
      if type(input_dict[key]) is not tu.Tube and type(input_dict[key]) is not Empty:
        self.slots[key].set_val(input_dict[key])
        self.slots[key].set_plug(input_dict[key], obj_is_callable=callable(input_dict[key]))
        self.slots[key].tube = empty

    if all_slots_filled:
      input_dict = self._convert_tubes_to_vals(input_dict)
      self.pour(**input_dict)

    import wtrwrks.tanks.clone as cl
    if isinstance(self, cl.MergeEqual):
      self._handle_merge()

    if self.pass_through_keys is not None:
      import wtrwrks.tanks.tank_defs as td
      for key in self.pass_through_keys:
        slot = self.slots[key]
        if type(slot.tube) is not Empty:
          td.merge_equal(slot.tube, self.tubes[key])
        if slot.plug is not None:
          self.tubes[key].plug = slot.plug

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
    input_dict : dict(
        keys - Slot keys. Must be the same as the attribute slot_keys.
        values - Tube, None or some valid input data type.
      )
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
      if type(input_dict[key]) is Empty:
        all_slots_filled = False
        break
      if (
        type(input_dict[key]) is tu.Tube and
        input_dict[key].get_val() is None):
        all_slots_filled = False
        break
      if type(input_dict[key]) is list or type(input_dict[key]) is tuple:
        break_out = False
        for val in input_dict[key]:
          if type(val) is Empty:
            all_slots_filled = False
            break_out = True
          if (
            type(val) is tu.Tube and
            val.get_val() is None):
            all_slots_filled = False
            break_out = True
            break
        if break_out:
          break
    return all_slots_filled

  def _convert_tubes_to_vals(self, input_dict):
    """Pull out the values associated with the tubes connected to this tank's slots. Where the values of the dictionary are the values stored in the tube, rather than the Tube object itself.

    Parameters
    ----------
    input_dict : dict(
        keys - Slot keys. Must be the same as the attribute slot_keys.
        values - Tube, None or some valid input data type.
      )
      The inputs to the tank.

    Returns
    -------
    dict(
      keys - Slot keys. Must be the same as the attribute slot_keys.
      values - valid input data type.
    )
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

  def _handle_iterable(self, iterable):
    found_tube = False
    for val in iterable:
      if type(val) is sl.Slot:
        raise ValueError("Cannot pass slot as argument to " + str(type(self)))

      if type(val) is tu.Tube or type(val) is Empty:
        found_tube = True
        break

    if not found_tube:
      return iterable

    import wtrwrks.tanks.tank_defs as td
    l_tubes, l_slots = td.tube_list(*iterable)

    for i, val in enumerate(iterable):
      if type(val) is not tu.Tube and type(val) is not Empty:
        l_slots['a' + str(i)].set_plug(val)

    return l_tubes['target']

  def _handle_merge(self):
    for key in self.slots:
      slot = self.slots[key]

      if type(slot.tube) is not Empty:
        slot.tube.downstream_tube = self.tubes['target']

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
    input_dict : dict(
        keys - Slot keys. Must be the same as the attribute slot_keys.
        values - Tube, None or some valid input data type.
      )
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

      # If the tube was already used for another tank, then it'll have to be
      # cloned.
      if type(tube.slot) is not Empty:

        # Save the slot in order to connect it to the clone tube later.
        other_slot = tube.slot
        tube.slot = empty

        import wtrwrks.tanks.tank_defs as td
        c_tubes, c_slots = td.clone(a=tube)

        # Join the other slot to the 'b' tube of the clone tank
        other_slot.tube = c_tubes['b']
        c_tubes['b'].slot = other_slot

        # Join this slot to the 'a' tube of the clone tank
        slot.tube = c_tubes['a']
        c_tubes['a'].slot = slot

        # Remove the newly created clone tupes from the taps, since they are
        # immediately connected to slots.
        del waterwork.taps[c_tubes['a'].name]
        del waterwork.taps[c_tubes['b'].name]

        # Remove the slots from the funnels, since the clone slot is now the
        # funnel.
        if slot.name in waterwork.funnels:
          del waterwork.funnels[slot.name]
          if slot.plug is not None:
            raise ValueError(str(slot) + ' was plugged but is no longer a funnel. Only funnels can have plugs.')
        if other_slot.name in waterwork.funnels:
          del waterwork.funnels[other_slot.name]
          if other_slot.plug is not None:
            raise ValueError(str(other_slot) + ' was plugged but is no longer a funnel. Only funnels can have plugs.')
      else:
        tube.slot = slot
        slot.tube = tube

      if type(tube) is tu.Tube:
        if tube.name in waterwork.taps:
          del waterwork.taps[tube.name]
        if slot.name in waterwork.funnels:
          del waterwork.funnels[slot.name]
        if slot.plug is not None:
          raise ValueError(str(slot) + ' was plugged but is no longer a funnel. Only funnels can have plugs.')
        if tube.plug is not None:
          raise ValueError(str(tube) + ' was plugged but is no longer a tap. Only taps can have plugs.')

  def _pour(self, *args, **kwargs):
    """Make sure pour function is set by subclass. The forward transformation of inputs."""
    raise ValueError("'_pour' method not defined for " + str(type(self)))

  def _pump(self, *args, **kwargs):
    """Make sure pump function is set by subclass. The backward transformation of outputs."""
    raise ValueError("'_pump' method not defined for " + str(type(self)))

  def _save_dict(self):
    save_dict = {}
    save_dict['func_name'] = self.func_name
    save_dict['name'] = self.name

    return save_dict

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
        keys - Tube keys. The same as the keys from attribute tube_keys.
        values - The data_types outputted by the tank.
      }
        All of the ouputs the tank gives in the 'pour' (i.e. forward) direction.

    """
    # Check that the inputs are valid
    if set(input_dict.keys()) != set(self.slot_keys):
      raise ValueError("Must pass " + str(input_dict.keys()) + " as arguments, got " + str(self.slot_keys))

    for key, val in input_dict.iteritems():
      if not self._slot_is_valid_type(key, val):
        raise TypeError("Got invalid type for (tank, slot): " + str((self.name, key)) + ". ")
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
        keys - Tube keys. Must be the same as the keys from attribute tube_keys.
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

    for key, val in kwargs.iteritems():
      if not self._tube_is_valid_type(key, val):
        raise TypeError("Got invalid type for (tank, tube): " + str((self.name, key)) + ". ")
    # Run the function defined by the subclass
    slot_dict = self._pump(**kwargs)

    # Set the vals
    for key in slot_dict:
      self.slots[key].set_val(slot_dict[key])

    return slot_dict

  def get_slot_tanks(self):
    """Get a set of all the tanks that pour into this one.

    Returns
    -------
    set of Tanks
      The set of all tanks that feed into this one (in the pour direction).

    """
    tanks = set()
    for slot_key in self.slots:
      slot = self.slots[slot_key]
      if type(slot.tube) is not Empty:
        tanks.add(slot.tube.tank)
    return tanks

  def get_tube_tanks(self):
    """Get a set of all the tanks that pump into this one.

    Returns
    -------
    set of Tanks
      The set of all tanks that feed into this one (in the pump direction).

    """
    tanks = set()
    for tube_key in self.tubes:
      tube = self.tubes[tube_key]
      if type(tube.slot) is not Empty:
        tanks.add(tube.slot.tank)
    return tanks

  def get_pour_dependencies(self):
    """Get a set of tanks that need to be run before this tank is run (in the pour direction).

    Returns
    -------
    set of Tanks
      The set of tanks that need to be run before this one when executing a pour of a Waterwork.

    """
    tanks = [self]
    dependencies = set()

    while tanks:
      tank = tanks.pop()

      # Go through each of the slots of the current tank, to get to the
      # 'parent_tank', i.e. the tank that has outputs which are needed for the
      # current tanks inputs. Put the parent tank before the current one.
      for slot_name in tank.slots:

        slot = tank.slots[slot_name]

        # If the slot is not connected to any tube (i.e. is a funnel) continue.
        # print slot.name, slot.name in slot.waterwork.funnels
        if slot.name in slot.waterwork.funnels:
          continue

        parent_tank = slot.tube.tank

        tanks.append(parent_tank)
        dependencies.add(parent_tank)

    return dependencies

  def get_pump_dependencies(self):
    """Get a set of tanks that need to be run before this tank is run (in the pump direction).

    Returns
    -------
    set of Tanks
      The set of tanks that need to be run before this one when executing a pump of a Waterwork.

    """
    tanks = [self]
    dependencies = set()
    while tanks:
      tank = tanks.pop()

      # Go through each of the slots of the current tank, to get to the
      # 'parent_tank', i.e. the tank that has outputs which are needed for the
      # current tanks inputs. Put the parent tank before the current one.
      for tube_name in tank.tubes:
        tube = tank.tubes[tube_name]

        # If the tube is not connected to any slot (i.e. is a tap) continue.
        if tube.name in tube.waterwork.taps:
          continue

        parent_tank = tube.slot.tank

        tanks.append(parent_tank)
        dependencies.add(parent_tank)
    return dependencies

  def _slot_is_valid_type(self, key, val):
    return True

  def _tube_is_valid_type(self, key, val):
    return True
