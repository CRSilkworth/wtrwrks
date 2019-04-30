import reversible_transforms.waterworks.waterwork_part as wp
import reversible_transforms.waterworks.tank as ta
import reversible_transforms.tanks.utils as ut
import numpy as np


def replace(a, mask, replace_with, type_dict=None, waterwork=None, name=None):
  """Find the min of a np.array along one or more axes in a reversible manner.

  Parameters
  ----------
  a : Tube, np.ndarray or None
      The array to get the min of.
  axis : Tube, int, tuple or None
      The axis (axes) along which to take the min.
  type_dict : dict({
    keys - ['a', 'b']
    values - type of argument 'a' type of argument 'b'.
  })
    The types of data which will be passed to each argument. Needed when the types of the inputs cannot be infered from the arguments a and b (e.g. when they are None).

  waterwork : Waterwork or None
    The waterwork to add the tank (operation) to. Default's to the _default_waterwork.
  name : str or None
      The name of the tank (operation) within the waterwork

  Returns
  -------
  Tank
      The created add tank (operation) object.

  """
  type_dict = ut.infer_types(type_dict, a=a, mask=mask, replace_with=replace_with)

  return Replace(a=a, mask=mask, replace_with=replace_with, waterwork=waterwork, name=name)


class Replace(ta.Tank):
  """The min class. Handles 'a's of np.ndarray type.

  Attributes
  ----------
  slot_keys : list of str
    The tank's (operation's) argument keys. They define the names of the inputs to the tank.
  tube_dict : dict(
    keys - strs. The tank's (operation's) output keys. THey define the names of the outputs of the tank
    values - types. The types of the arguments outputs.
  )
    The tank's (operation's) output keys and their corresponding types.

  """

  slot_keys = ['a', 'mask', 'replace_with']
  tube_dict = {
    'target': np.ndarray,
    'replaced_vals': np.ndarray,
    'mask': int,
    'replace_with_shape': tuple
  }

  def _pour(self, a, mask, replace_with):
    """Execute the add in the pour (forward) direction .

    Parameters
    ----------
    a : np.ndarray
      The array to take the min over.
    axis : int, tuple
      The axis (axes) to take the min over.

    Returns
    -------
    dict(
      'target': np.ndarray
        The result of the min operation.
      'a': np.ndarray
        The original a
      'axis': in, tuple
        The axis (axes) to take the min over.
    )

    """
    target = ut.maybe_copy(a)
    replaced_vals = target[mask]
    target[mask] = replace_with
    # Must just return 'a' as well since so much information is lost in a
    # min
    return {'target': target, 'mask': mask, 'replaced_vals': replaced_vals.flatten(), 'replace_with_shape': replace_with.shape}

  def _pump(self, target, mask, replaced_vals, replace_with_shape):
    """Execute the add in the pump (backward) direction .

    Parameters
    ----------
    target: np.ndarray
      The result of the min operation.
    a : np.ndarray
      The array to take the min over.
    axis : int, tuple
      The axis (axes) to take the min over.

    Returns
    -------
    dict(
      'a': np.ndarray
        The original a
      'axis': in, tuple
        The axis (axes) to take the min over.
    )

    """
    a = ut.maybe_copy(target)
    replace_with = a[mask]

    masked_shape = a[mask].shape
    a[mask] = replaced_vals.reshape(masked_shape)

    if replace_with_shape:
      num_elements = np.prod(replace_with_shape)
    else:
      num_elements = 1

    if num_elements == 1:
      replace_with = replace_with.flatten()[0].reshape(replace_with_shape)

    return {'a': a, 'mask': mask, 'replace_with': replace_with}
