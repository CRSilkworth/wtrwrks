import wtrwrks.waterworks.globs as gl
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.name_space as ns
from wtrwrks.waterworks.empty import empty
import os
import pprint


class Waterwork(object):
  """The full graph of tanks (i.e. operations) on the data, along with all slots and tubes which define the inputs/outputs of operations and hold their values. Can be thought of as a larger reversible operation that are composed of many smaller reversible operations.

  Attributes
  ----------
  funnels : dict(
    keys - strs. Names of the funnels.
    values - Slot objects.
  )
    All of the slots defined within the waterwork which are not connected to some other tube. i.e. the 'open' slots that need data in order to produce an output in the pour direction.
  taps : dict(
    keys - strs. Names of the taps.
    values - Tube objects.
  )
    All of the tubes defined within the waterwork which are not connected to some other slot. i.e. the 'open' tubes that need data in order to produce an output in the pump direction.
  slots : dict(
    keys - strs. Names of the slots.
    values - Slot objects.
  )
    All of the slots defined within the waterwork.
  tubes : dict(
    keys - strs. Names of the tubes.
    values - Tube objects.
  )
    All of the tubes defined within the waterwork.
  tanks : dict(
    keys - strs. Names of the tanks.
    values - Tube objects.
  )
    All of the tanks (or operations) defined within the waterwork.
  """

  def __init__(self, name=''):
    """Initialize the waterwork to have empty funnels, slots, tanks, and taps."""
    self.funnels = {}
    self.tubes = {}
    self.slots = {}
    self.tanks = {}
    self.taps = {}
    self.name = name

  def __enter__(self):
    """When entering, set the global _default_waterwork to this waterwork."""
    if gl._default_waterwork is not None:
      raise ValueError("_default_waterwork is already set. Cannot be reset until context is exitted. Are you within the with statement of another waterwork?")

    # Create a new namespace for this waterwork
    self.name_space = ns.NameSpace(self.name)
    self.name_space.__enter__()

    gl._default_waterwork = self
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """When exiting, set the global _default_waterwork back to None."""
    gl._default_waterwork = None
    self.name_space.__exit__(exc_type, exc_val, exc_tb)

  def _pour_tank_order(self):
    """Get the order to calculate the tanks in the pour direction.

    Returns
    -------
    list of tank objects
        The tanks ordered in such a way that they are guaranteed to have all the information to perform the operation.

    """
    tanks = sorted([self.tanks[t] for t in self.tanks if not self.tanks[t].get_slot_tanks()])
    visited = set(tanks)
    sorted_tanks = []

    while tanks:
      tank = tanks.pop(0)
      sorted_tanks.append(tank)

      child_tanks = tank.get_tube_tanks() - visited
      for child_tank in child_tanks:
        # If there are any tanks child_tank depnends on , that haven't been
        # visited, then continue.
        if child_tank.get_slot_tanks() - visited:
          continue

        tanks.append(child_tank)
        visited.add(child_tank)

    return sorted_tanks

  def _pump_tank_order(self):
    """Get the order to calculate the tanks in the pump direction.

    Returns
    -------
    list of tank objects
        The tanks ordered in such a way that they are guaranteed to have all the information to perform the operation.

    """
    tanks = sorted([self.tanks[t] for t in self.tanks if not self.tanks[t].get_tube_tanks()])
    visited = set(tanks)
    sorted_tanks = []

    while tanks:
      tank = tanks.pop(0)
      sorted_tanks.append(tank)

      child_tanks = tank.get_slot_tanks() - visited
      for child_tank in child_tanks:
        # If there are any tanks child_tank depnends on , that haven't been
        # visited, then continue.
        if child_tank.get_tube_tanks() - visited:
          continue

        tanks.append(child_tank)
        visited.add(child_tank)
    # tanks = sorted([self.tanks[t] for t in self.tanks])
    # return sorted(tanks, cmp=lambda a, b: 0 if b in a.get_pump_dependencies() else -1)
    return sorted_tanks

  def maybe_get_slot(self, *args):
    """Get a particular tank's if it exists, otherwise return None. Can take a variety input types.

    Parameters
    ----------
    *args: Slot, str or tuple
      Either the slot itself, the name of the slot or the (tank name, slot key) tuple.

    Returns
    -------
    Slot or None
      The slot object if found, otherwise a None

    """
    import wtrwrks.waterworks.slot as sl
    # Pull out the tank and key depending on the type and number of inputs.
    if len(args) == 2:
      tank = args[0]
      key = args[1]
    elif len(args) == 1 and type(args[0]) is tuple:
      tank = args[0][0]
      key = args[0][1]
    elif len(args) == 1 and type(args[0]) in (str, unicode) and args[0] in self.slots:
      return self.slots[args[0]]
    elif len(args) == 1 and isinstance(args[0], sl.Slot):
      return args[0]
    else:
      return None

    # Pull out the relevant tank object.
    if type(tank) in (str, unicode) and tank in self.tanks:
      tank = self.tanks[tank]
    elif isinstance(tank, sl.Slot):
      pass
    else:
      return None

    # Get the slot
    if key in tank.slots:
      return tank.slots[key]

    return None

  def maybe_get_tube(self, *args):
    """Get a particular tank's if it exists, otherwise return None. Can take a variety input types.

    Parameters
    ----------
    *args: tube, str or tuple
      Either the tube itself, the name of the tube or the (tank name, tube key) tuple.

    Returns
    -------
    tube or None
      The tube object if found, otherwise a None

    """
    import wtrwrks.waterworks.tube as tu
    # Pull out the tank and key depending on the type and number of inputs.
    if len(args) == 2:
      tank = args[0]
      key = args[1]
    elif len(args) == 1 and type(args[0]) is tuple:
      tank = args[0][0]
      key = args[0][1]
    elif len(args) == 1 and type(args[0]) in (str, unicode) and args[0] in self.tubes:
      return self.tubes[args[0]]
    elif len(args) == 1 and isinstance(args[0], tu.Tube):
      return args[0]
    else:
      return None

    # Pull out the relevant tank object.
    if type(tank) in (str, unicode) and tank in self.tanks:
      tank = self.tanks[tank]
    elif isinstance(tank, tu.Tube):
      pass
    else:
      return None

    # Get the tube
    if key in tank.tubes:
      return tank.tubes[key]

    return None

  def get_slot(self, tank, key):
    """Get a particular tank's slot.

    Parameters
    ----------
    tank : Tank or str
        Either the tank object or the name of the tank.
    key : str
        The slot key of the slot for that tank.

    Returns
    -------
    Slot
        The slot object

    """
    if type(tank) in (str, unicode):
      tank = self.tanks[tank]

    return self.tanks[tank.name].slots[key]

  def get_tube(self, tank, key):
    """Get a particular tank's tube.

    Parameters
    ----------
    tank : Tank or str
        Either the tank object or the name of the tank.
    key : str
        The tube key of the tube for that tank.

    Returns
    -------
    Tube
        The tube object.

    """
    if type(tank) in (str, unicode):
      tank = self.tanks[tank]

    return self.tanks[tank.name].tubes[key]

  def pour(self, funnel_dict=None, key_type='tube'):
    """Run all the operations of the waterwork in the pour (or forward) direction.

    Parameters
    ----------
    funnel_dict : dict(
      keys - Slot objects or Placeholder objects. The 'funnels' (i.e. unconnected slots) of the waterwork.
      values - valid input data types
    )
        The inputs to the waterwork's full pour function.
    key_type : str ('tube', 'tuple', 'name')
      The type of keys to return in the return dictionary. Can either be the tube objects themselves (tube), the tank, output key pair (tuple) or the name (str) of the tube.

    Returns
    -------
    dict(
      keys - Tube objects, (or tuples if tuple_keys set to True). The 'taps' (i.e. unconnected tubes) of the waterwork.
    )
        The outputs of the waterwork's full pour function

    """
    if funnel_dict is None:
      funnel_dict = {}
    # Set all the values of the funnels from the inputted arguments.
    for ph, val in funnel_dict.iteritems():
      sl_obj = self.maybe_get_slot(ph)
      if sl_obj is not None:
        sl_obj.set_val(val)
        if sl_obj.tube is not empty:
          sl_obj.tube.set_val(val)
      else:
        raise ValueError(str(ph) + ' is not a supported input into pour function')

    # Check that all funnels have a value
    for funnel in self.funnels:
      if self.funnels[funnel].get_val() is None:
        raise ValueError("All funnels must have a set value. " + str(funnel) + " is not set.")

    # Run all the tanks (operations) in the pour direction, filling all slots'
    # and tubes' val attributes as you go.
    tanks = self._pour_tank_order()
    for tank in tanks:
      kwargs = {k: tank.slots[k].get_val() for k in tank.slots}
      tube_dict = tank.pour(**kwargs)

      for key in tube_dict:
        slot = tank.tubes[key].slot

        if slot is not empty:
          slot.set_val(tube_dict[key])

    # Create the dictionary to return
    r_dict = {}
    for tap_name in self.taps:
      tap = self.taps[tap_name]
      if key_type == 'tube':
        r_dict[tap] = tap.get_val()
      elif key_type == 'tuple':
        r_dict[tap.get_tuple()] = tap.get_val()
      elif key_type == 'str':
        r_dict[tap.name] = tap.get_val()
      else:
        raise ValueError(str(key_type) + " is an invalid key_type.")

    return r_dict

  def pump(self, tap_dict=None, key_type='slot'):
    """Run all the operations of the waterwork in the pump (or backward) direction.

    Parameters
    ----------
    funnel_dict : dict(
      keys - Tube objects. The 'taps' (i.e. unconnected tubes) of the waterwork.
    )
        The inputs of the waterwork's full pump function
    key_type : str ('tube', 'tuple', 'name')
      The type of keys to return in the return dictionary. Can either be the tube objects themselves (tube), the tank, output key pair (tuple) or the name (str) of the tube.

    Returns
    -------
    dict(
      keys - Slot objects. The 'funnels' (i.e. unconnected slots) of the waterwork.
      values - valid input data types
    )
        The outputs to the waterwork's full pump function.

    """
    if tap_dict is None:
      tap_dict = {}
    # Set all the values of the taps from the inputted arguments.
    for tap, val in tap_dict.iteritems():
      tu_obj = self.maybe_get_tube(tap)
      if tu_obj is not None:
        tu_obj.set_val(val)
      else:
        raise ValueError(str(tap) + ' is not a supported form of input into pump function')

    # Check that all funnels have a value
    for tap in self.taps:
      if self.taps[tap].get_val() is None:
        raise ValueError("All taps must have a set value. " + str(tap) + " is not set.")

    # Run all the tanks (operations) in the pump direction, filling all slots'
    # and tubes' val attributes as you go.
    tanks = self._pump_tank_order()
    for tank in tanks:
      kwargs = {k: tank.tubes[k].get_val() for k in tank.tubes}
      slot_dict = tank.pump(**kwargs)

      for key in slot_dict:
        tube = tank.slots[key].tube

        if tube is not empty:
          tube.set_val(slot_dict[key])

    # Create the dictionary to return
    r_dict = {}
    for funnel_name in self.funnels:
      funnel = self.funnels[funnel_name]
      if key_type == 'slot':
        r_dict[funnel] = funnel.get_val()
      elif key_type == 'tuple':
        r_dict[funnel.get_tuple()] = funnel.get_val()
      elif key_type == 'str':
        r_dict[funnel.name] = funnel.get_val()
      else:
        raise ValueError(str(key_type) + " is an invalid key_type.")

    return r_dict

  def clear_vals(self):
    """Set all the slots, tubes and placeholder values back to None """
    for d in [self.slots, self.tubes]:
      for key in d:
        d[key].set_val(None)
