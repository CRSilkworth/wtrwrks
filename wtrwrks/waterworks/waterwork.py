import wtrwrks.waterworks.globs as gl
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.name_space as ns
import wtrwrks.utils.dir_functions as d
from wtrwrks.waterworks.empty import empty
import wtrwrks.read_write.tf_features as feat
import os
import pprint
import importlib
import dill as pickle
import tensorflow as tf
import numpy as np
import logging


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

  def __init__(self, name='', from_file=None):
    """Initialize the waterwork to have empty funnels, slots, tanks, and taps."""
    self.funnels = {}
    self.tubes = {}
    self.slots = {}
    self.tanks = {}
    self.taps = {}
    self.name = name

    if from_file is not None:
      save_dict = d.read_from_file(from_file)
      self._from_save_dict(save_dict)

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

  def _from_save_dict(self, save_dict):
    import wtrwrks.tanks.tank_defs as td
    with ns.NameSpace(''):
      self.name = save_dict['name']
      for tank_name in save_dict['tanks']:
        tank_dict = save_dict['tanks'][tank_name]
        func = getattr(td, tank_dict['func_name'])

        kwargs = {}
        if 'kwargs' in tank_dict:
          kwargs = tank_dict['kwargs']

        tubes, slots = func(name=tank_name, waterwork=self, **kwargs)
        tank = tubes[tubes.keys()[0]].tank
        self.tanks[tank_name] = tank

      for slot_name in save_dict['slots']:
        slot_dict = save_dict['slots'][slot_name]
        tank = self.tanks[slot_dict['tank']]

        slot = tank.get_slot(slot_dict['key'])
        slot.plug = slot_dict['plug']

        # Set to proper name
        del self.slots[slot.name]
        slot.name = slot_name
        self.slots[slot_name] = slot

      for tube_name in save_dict['tubes']:
        tube_dict = save_dict['tubes'][tube_name]
        tank = self.tanks[tube_dict['tank']]

        tube = tank.get_tube(tube_dict['key'])
        tube.plug = tube_dict['plug']

        # Set to proper name
        del self.tubes[tube.name]
        tube.name = tube_name
        self.tubes[tube_name] = tube

      for slot_name in self.slots:
        slot_dict = save_dict['slots'][slot_name]
        slot = self.slots[slot_name]

        if slot_dict['tube'] is not None:
          tube = self.tubes[slot_dict['tube']]
          tube.slot = slot
        else:
          tube = empty

        slot.tube = tube

      self.funnels = {}
      for funnel_name in save_dict['funnels']:
        self.funnels[funnel_name] = self.slots[funnel_name]

      self.taps = {}
      for tap_name in save_dict['taps']:
        self.taps[tap_name] = self.tubes[tap_name]

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

  def _save_dict(self):
    save_dict = {}
    save_dict['name'] = self.name
    save_dict['funnels'] = sorted(self.funnels.keys())
    save_dict['taps'] = sorted(self.taps.keys())

    save_dict['tanks'] = {}
    save_dict['slots'] = {}
    save_dict['tubes'] = {}

    for key in self.tanks:
      tank = self.tanks[key]
      save_dict['tanks'][key] = tank._save_dict()

    for key in self.slots:
      slot = self.slots[key]
      save_dict['slots'][key] = slot._save_dict()

    for key in self.tubes:
      tube = self.tubes[key]
      save_dict['tubes'][key] = tube._save_dict()

    return save_dict

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

  def get_taps(self, return_plugged=False):
    """Get a dictionary of all the taps. Optionally choose to return the taps which are plugged or not."""
    r_dict = {k: v for k, v in self.taps.iteritems() if v.plug is None}
    return r_dict

  def get_funnels(self, return_plugged=False):
    """Get a dictionary of all the funnels. Optionally choose to return the funnels which are plugged or not."""
    r_dict = {k: v for k, v in self.funnels.iteritems() if v.plug is None}
    return r_dict

  def pour(self, funnel_dict=None, key_type='tube', return_plugged=False):
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
    stand_funnel_dict = {}
    logging.debug("%s's funnel_dict - %s", self.name, sorted(funnel_dict))
    for ph, val in funnel_dict.iteritems():
      sl_obj = self.maybe_get_slot(ph)
      if sl_obj is not None:
        if sl_obj.plug is not None:
          raise ValueError(str(sl_obj) + ' has a plug. If you want to set the value dynamically then do funnel.unplug().')

        stand_funnel_dict[sl_obj.name] = val
        sl_obj.set_val(val)
        if sl_obj.tube is not empty:
          sl_obj.tube.set_val(val)
      else:
        raise ValueError(str(ph) + ' is not a supported input into pour function')

    # Check that all funnels have a value
    logging.debug("%s's funnels - %s", self.name, sorted(self.funnels))
    for funnel_key in self.funnels:
      logging.debug("funnel_key - %s", funnel_key)
      funnel = self.funnels[funnel_key]
      if funnel.plug is not None:
        logging.debug("Plugging - %s", funnel_key)
        funnel.set_val(funnel.plug(stand_funnel_dict))
      elif funnel.get_val() is None:
        raise ValueError("All funnels must have a set value. " + str(funnel) + " is not set.")

    # Run all the tanks (operations) in the pour direction, filling all slots'
    # and tubes' val attributes as you go.
    tanks = self._pour_tank_order()
    logging.debug("%s's pour_tank_order - %s", self.name, [t.name for t in tanks])
    for tank in tanks:
      logging.info("Pouring tank - %s", tank.name)
      kwargs = {k: tank.slots[k].get_val() for k in tank.slots}

      logging.debug("Inputting kwargs to pour - %s", {k: v for k, v in kwargs.iteritems()})
      tube_dict = tank.pour(**kwargs)

      for key in tube_dict:
        slot = tank.tubes[key].slot

        if slot is not empty:
          slot.set_val(tube_dict[key])

    # Create the dictionary to return
    r_dict = {}
    logging.debug("%s's taps - %s", self.name, sorted(self.taps))
    for tap_name in self.taps:
      logging.debug("setting tap_key - %s", tap_name)
      tap = self.taps[tap_name]

      if tap.plug is not None and not return_plugged:
        continue

      if key_type == 'tube':
        r_dict[tap] = tap.get_val()
      elif key_type == 'tuple':
        r_dict[tap.get_tuple()] = tap.get_val()
      elif key_type == 'str':
        r_dict[tap.name] = tap.get_val()
      else:
        raise ValueError(str(key_type) + " is an invalid key_type.")

    return r_dict

  def pump(self, tap_dict=None, key_type='slot', return_plugged=False):
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

    stand_tap_dict = {}
    # Set all the values of the taps from the inputted arguments.
    logging.debug("%s's tap_dict - %s", self.name, sorted(tap_dict))
    for tap, val in tap_dict.iteritems():
      tu_obj = self.maybe_get_tube(tap)
      if tu_obj is not None:
        if tu_obj.plug is not None:
          raise ValueError(str(tu_obj) + ' has a plug. Cannot set the value of a funnel with a plug.')

        stand_tap_dict[tu_obj.name] = val
        tu_obj.set_val(val)
      else:
        raise ValueError(str(tap) + ' is not a supported form of input into pump function')

    # Check that all funnels have a value
    logging.debug("%s's taps - %s", self.name, sorted(self.taps))
    for tap_key in self.taps:
      logging.debug("tap_key - %s", tap_key)
      tap = self.taps[tap_key]
      if tap.plug is not None:
        logging.debug("Plugging - %s", tap_key)
        tap.set_val(tap.plug(stand_tap_dict))
      elif tap.get_val() is None:
        raise ValueError("All taps must have a set value. " + str(tap) + " is not set.")

    # Run all the tanks (operations) in the pump direction, filling all slots'
    # and tubes' val attributes as you go.
    tanks = self._pump_tank_order()
    logging.debug("%s's pump_tank_order - %s", self.name, [t.name for t in tanks])
    for tank in tanks:
      logging.info("Pumping tank - %s", tank.name)
      kwargs = {k: tank.tubes[k].get_val() for k in tank.tubes}

      logging.debug("Inputting kwargs to pour - %s", {k: v for k, v in kwargs.iteritems()})
      slot_dict = tank.pump(**kwargs)

      for key in slot_dict:
        tube = tank.slots[key].tube

        if tube is not empty:
          tube.set_val(slot_dict[key])

    # Create the dictionary to return
    r_dict = {}
    logging.debug("%s's funnels - %s", self.name, sorted(self.funnels))
    for funnel_name in self.funnels:
      funnel = self.funnels[funnel_name]

      if funnel.plug is not None and not return_plugged:
        continue

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

  def save_to_file(self, file_name):
    if not file_name.endswith('pickle') and not file_name.endswith('pkl') and not file_name.endswith('dill'):
      raise ValueError("Waterwork can only be saved as a pickle.")
    save_dict = self._save_dict()
    d.save_to_file(save_dict, file_name)

  def write_examples(self, array, file_name):
    """Pours the array then writes the examples to tfrecords. It creates one example per 'row', i.e. axis=0 of the arrays. All arrays must have the same axis=0 dimension and must be of a type that can be written to a tfrecord

    Parameters
    ----------
    array : np.ndarray
      The array to transform to examples, then write to disk.
    file_name : str
      The name of the tfrecord file to write to.

    """
    writer = tf.python_io.TFRecordWriter(file_name)
    tap_dict = self.pour(array)

    att_dict = {}

    num_examples = None
    for key in tap_dict:
      array = tap_dict[key]

      if num_examples is None:
        num_examples = array.shape[0]

      if array.shape[0] != num_examples:
        raise ValueError("All arrays must have the same size first dimesion in order to split them up into individual examples")

      if array.dtype not in (np.int32, np.int64, np.bool, np.float32, np.float64) and array.dtype.type not in (np.string_, np.unicode_):
        raise TypeError("Only string and number types are supported. Got " + str(array.dtype))

      att_dict[key]['dtype'] = str(array.dtype)
      att_dict[key]['shape'] = list(array.shape[1:])
      att_dict[key]['size'] = np.prod(att_dict[key]['shape'])
      att_dict[key]['feature_func'] = feat.select_feature_func(array.dtype)

    for row_num in xrange(num_examples):
      example_dict = {}
      for key in tap_dict:
        flat = tap_dict[key][row_num].flatten()

        example_dict[os.path.join(key, 'vals')] = att_dict[key]['feature_func'](flat)

        example_dict[os.path.join(key, 'shape')] = feat._int_feat(att_dict[key]['shape'])

      example = tf.train.Example(
        features=tf.train.Features(feature=example_dict)
      )
      writer.write(example.SerializeToString())

    writer.close()

  def read_and_decode(self, serialized_example, feature_dict, prefix=''):
    """Convert a serialized example created from an example dictionary from this transform into a dictionary of shaped tensors for a tensorflow pipeline.

    Parameters
    ----------
    serialized_example : tfrecord serialized example
      The serialized example to read and convert to a dictionary of tensors.
    feature_dict :
      A dictionary with the keys being the keys of the tap dict (and their shapes) and the values being the FixedLenFeature's with the proper values all filled in

    Returns
    -------
    dict of tensors
      The tensors created by decoding the serialized example and reshaping them.

    """
    features = tf.parse_single_example(
      serialized_example,
      features=feature_dict
    )

    for key in features:
      if key.endswith('/shape'):
        continue
      features[key] = tf.reshape(features[key], features[os.path.join(key, 'shape')])

    return features

  def parse_examples(self, tap_dict, dtype_dict=None, key_type='slot'):
    """Run the pump transformation on a list of example dictionaries to reconstruct the original array.

    Parameters
    ----------
    batched_arrays: dict of arrays
      The keys

    Returns
    -------
    np.ndarray
      The numpy array to transform into examples.

    """
    for key in dtype_dict:
      tap_dict[key] = tap_dict[key].astype(dtype_dict[key])

    return self.pump(tap_dict, key_type=key_type)
