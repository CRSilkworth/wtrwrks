import reversible_transforms.waterworks.globs as gl
import reversible_transforms.waterworks.waterwork_part as wp
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
    self.placeholders = {}
    self.name = name

  def __enter__(self):
    """When entering, set the global _default_waterwork to this waterwork."""
    if gl._default_waterwork is not None:
      raise ValueError("_default_waterwork is already set. Cannot be reset until context is exitted. Are you within the with statement of another waterwork?")
    gl._default_waterwork = self
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """When exiting, set the global _default_waterwork back to None."""
    gl._default_waterwork = None

  def _pour_tank_order(self):
    """Get the order to calculate the tanks in the pour direction.

    Returns
    -------
    list of tank objects
        The tanks ordered in such a way that they are guaranteed to have all the information to perform the operation.

    """
    tanks = sorted([self.tanks[t] for t in self.tanks])
    return sorted(tanks, cmp=lambda a, b: 1 if b in a.get_pour_dependencies() else -1)

  def _pump_tank_order(self):
    """Get the order to calculate the tanks in the pump direction.

    Returns
    -------
    list of tank objects
        The tanks ordered in such a way that they are guaranteed to have all the information to perform the operation.

    """
    tanks = sorted([self.tanks[t] for t in self.tanks])
    return sorted(tanks, cmp=lambda a, b: 0 if b in a.get_pump_dependencies() else -1)

  def _sorted_tap_names(self):
    """Sort all the taps in such a way that the taps corresponding to tanks that have none of their tubes being consumed by another tank appear first."""
    def sort_key(k):
      return len(self.taps[k].tank.paired_tubes())
    return sorted(self.taps, key=sort_key)

  def _sorted_funnel_names(self):
    """Sort all the funnels in such a way that the funnels corresponding to tanks that have none of their slots being filled by another tank's tube appear first."""
    def sort_key(k):
      return len(self.funnels[k].tank.paired_slots())
    return sorted(self.funnels, key=sort_key)

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

    return self.slots[str((tank.name, key))]

  def get_tube(self, tank, key):
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
        The slot object.

    """
    if type(tank) in (str, unicode):
      tank = self.tanks[tank]

    return self.tubes[str((tank.name, key))]

  def merge(self, other, join_dict, name='merged'):
    """Create a new waterwork by merging other into self(in the pour direction).

    Parameters
    ----------
    other : waterwork
        The waterwork to merge with self to create a new waterwork object.
    join_dict : dict(
      keys - slots from other
      values - tubes from self
    )
        The dictionary that describes the connections between self and other.
    name : str
        The name of the new waterwork to be created.

    Returns
    -------
    waterwork
        The waterwork formed by merging self and other together.

    """
    import reversible_transforms.waterworks.placeholder as pl

    if self.name == other.name:
      raise ValueError("Cannot merge two waterworks with the same name.")

    with Waterwork(name=name) as ww:
      for slot in join_dict:
        tube = join_dict[slot]
        slot.tube = tube
        tube.slot = slot

        del self.taps[tube.name]

        if type(tube) is not pl.Placeholder:
          del other.funnels[slot.name]

      # Go throuh each self's tanks and create a copy for the new waterwork
      for w in [self, other]:
        for d_name in ['funnels', 'taps', 'slots', 'tubes', 'tanks']:
          d = getattr(w, d_name)
          ww_d = getattr(ww, d_name)

          if d_name is 'tanks':
            new_d = {os.path.join(ww.name, k): v for k, v in d.iteritems()}
          else:
            new_d = {}
            for k, v in d.iteritems():
              tank_name = os.path.join(ww.name, v.tank.name)
              new_d[str((tank_name, v.key))] = v
          ww_d.update(new_d)

      all_tanks = self._pour_tank_order() + other._pour_tank_order()
      for tank in all_tanks:
        tank_name = os.path.join(ww.name, tank.name)
        for slot_key in tank.slots:
          slot = tank.slots[slot_key]
          slot.waterwork = ww
          slot.name = str((tank_name, slot.key))

        for tube_key in tank.tubes:
          tube = tank.tubes[tube_key]
          tube.waterwork = ww
          tube.name = str((tank_name, tube.key))

      for w in [self, other]:
        w.funnels = {}
        w.tubes = {}
        w.slots = {}
        w.tanks = {}
        w.taps = {}

    return ww

  def combine(self, other, join_dict, name='merged'):
    """Create a new waterwork by combining first self and then other (in the pour direction).

    Parameters
    ----------
    other : waterwork
        The waterwork to merge with self to create a new waterwork object.
    join_dict : dict(
      keys - slots from other
      values - tubes from self
    })
        The dictionary that describes the connections between self and other.
    name : str
        The name of the new waterwork to be created.

    Returns
    -------
    waterwork
        The waterwork formed by merging self and other together.

    """
    import reversible_transforms.waterworks.placeholder as pl

    if self.name == other.name:
      raise ValueError("Cannot merge two waterworks with the same name.")

    with Waterwork() as ww:
      # Go throuh each self's tanks and create a copy for the new waterwork
      tank_order = self._pour_tank_order()
      for tank in tank_order:

        # Create the input_dict to feed to the tank's constructor by taking
        # each of self's tank's slots, finding the corresponding tube (if
        # applicable) and creating a new tube with all the same parameters.
        input_dict = {}
        for slot_key in tank.slots:
          slot = tank.slots[slot_key]
          if type(slot.tube) is pl.Placeholder:
            input_dict[slot_key] = pl.Placeholder(
              val_type=slot.tube.val_type,
              val_dtype=slot.tube.val_dtype,
              val=slot.tube.val
            )
          else:
            parent_tank_name = os.path.join(name, slot.tube.tank.name)
            new_tube_name = str((parent_tank_name, slot.tube.key))
            input_dict[slot_key] = ww.tubes[new_tube_name]

        # Create the tank using input_dict defined above and then set all the
        # vals of the slots and tubes.
        cls = tank.__class__
        new_tank = cls(name=os.path.join(name, tank.name), **input_dict)
        for slot_key in tank.slots:
          new_tank.slots[slot_key].set_val(tank.slots[slot_key].val)
        for tube_key in tank.tubes:
          new_tank.tubes[tube_key].set_val(tank.tubes[tube_key].val)

      # Go throuh each other's tanks and create a copy for the new waterwork
      tank_order = other._pour_tank_order()
      for tank in tank_order:

        # Create the input_dict to feed to the tank's constructor by taking
        # each of other's tank's slots, finding the corresponding tube, whether it
        # be from within other or from self, as defined by join_dict (if
        # applicable) and creating a new tube with all the same parameters.
        input_dict = {}
        for slot_key in tank.slots:
          slot = tank.slots[slot_key]
          if slot in join_dict:
            parent_tank_name = os.path.join(name, join_dict[slot].tank.name)
            new_tube_name = str((parent_tank_name, join_dict[slot].key))
            input_dict[slot_key] = ww.tubes[new_tube_name]
          elif type(slot.tube) is pl.Placeholder:
            input_dict[slot_key] = pl.Placeholder(
              val_type=slot.tube.val_type,
              val_dtype=slot.tube.val_dtype,
              val=slot.tube.val
            )
          else:
            parent_tank_name = os.path.join(name, slot.tube.tank.name)
            new_tube_name = str((parent_tank_name, slot.tube.key))
            input_dict[slot_key] = ww.tubes[new_tube_name]

        # Create the tank using input_dict defined above and then set all the
        # vals of the slots and tubes.
        cls = tank.__class__
        new_tank = cls(name=os.path.join(name, tank.name), **input_dict)
        for slot_key in tank.slots:
          new_tank.slots[slot_key].set_val(tank.slots[slot_key].val)
        for tube_key in tank.tubes:
          new_tank.tubes[tube_key].set_val(tank.tubes[tube_key].val)

    return ww

  def pour(self, funnel_dict, tuple_keys=False):
    """Run all the operations of the waterwork in the pour(or forward) direction.

    Parameters
    ----------
    funnel_dict : dict(
      keys - Slot objects or Placeholder objects. The 'funnels' (i.e. unconnected slots) of the waterwork.
      values - valid input data types
    )
        The inputs to the waterwork's full pour function.
    tuple_keys : bool
      Whether or not the return dictionary should have tuples as keys rather than tubes.
    Returns
    -------
    dict(
      keys - Tube objects, (or tuples if tuple_keys set to True). The 'taps' (i.e. unconnected tubes) of the waterwork.
    )
        The outputs of the waterwork's full pour function

    """
    import reversible_transforms.waterworks.placeholder as pl

    # Set all the values of the funnels from the inputted arguments.
    for ph in funnel_dict:
      ph_obj = ph
      if type(ph) in (tuple, str, unicode) and str(ph) in self.placeholders:
        ph_obj = self.placeholders[str(ph)]
      elif type(ph) in (tuple, str, unicode) and str(ph) in self.funnels:
        ph_obj = self.funnels[str(ph)]
      elif isinstance(ph, wp.WaterworkPart):
        pass
      else:
        raise ValueError(str(type(ph)) + ' is not a supported form of input into ')

      ph_obj.set_val(funnel_dict[ph])
      if type(ph_obj) is pl.Placeholder and ph_obj.slot is not None:
        ph_obj.slot.set_val(funnel_dict[ph])

    # Check that all funnels have a value
    for funnel in self.funnels:
      if self.funnels[funnel].get_val() is None:
        raise ValueError("All funnels must have a set value. " + str(funnel) + " is not set.")

    # Run all the tanks (operations) in the pour direction, filling all slots'
    # and tubes' val attributes as you go.
    tanks = self._pour_tank_order()
    # print [str(t) for t in tanks]
    for tank in tanks:
      kwargs = {k: tank.slots[k].get_val() for k in tank.slots}
      tube_dict = tank.pour(**kwargs)

      for key in tube_dict:
        slot = tank.tubes[key].slot

        if slot is not None:
          slot.set_val(tube_dict[key])

    # Create the dictionary to return
    r_dict = {}
    for tap_name in self.taps:
      tap = self.taps[tap_name]
      if not tuple_keys:
        r_dict[tap] = tap.get_val()
      else:
        r_dict[tap.get_tuple()] = tap.get_val()

    return r_dict

  def pump(self, tap_dict, tuple_keys=False):
    """Run all the operations of the waterwork in the pump (or backward) direction.

    Parameters
    ----------
    funnel_dict : dict(
      keys - Tube objects. The 'taps' (i.e. unconnected tubes) of the waterwork.
    )
        The inputs of the waterwork's full pump function
    tuple_keys : bool
      Whether or not the return dictionary should have tuples as keys rather than tubes.
    Returns
    -------
    dict(
      keys - Slot objects. The 'funnels' (i.e. unconnected slots) of the waterwork.
      values - valid input data types
    )
        The outputs to the waterwork's full pump function.

    """
    # Set all the values of the taps from the inputted arguments.
    for tap in tap_dict:
      tap_obj = tap
      if type(tap) in (tuple, str, unicode):
        tap_obj = self.taps[str(tap)]
      tap_obj.set_val(tap_dict[tap])

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

        if tube is not None:
          tube.set_val(slot_dict[key])

    # Create the dictionary to return
    r_dict = {}
    for funnel_name in self.funnels:
      funnel = self.funnels[funnel_name]
      if not tuple_keys:
        r_dict[funnel] = funnel.get_val()
      else:
        r_dict[funnel.get_tuple()] = funnel.get_val()

    return r_dict

  def clear_vals(self):
    """Set all the slots, tubes and placeholder values back to None """
    for d in [self.slots, self.tubes, self.placeholders]:
      for key in d:
        d[key].set_val(None)
