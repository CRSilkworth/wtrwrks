"""Replace tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import wtrwrks.utils.array_functions as af
import numpy as np


class Replace(ta.Tank):
  """The defintion of the Replace tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'replace'
  slot_keys = ['a', 'mask', 'replace_with']
  tube_keys = ['target', 'mask', 'replaced_vals', 'replace_with']
  pass_through_keys = ['mask', 'replace_with']

  def _pour(self, a, mask, replace_with):
    """Execute the Replace tank (operation) in the pour (forward) direction.

    Parameters
    ----------
    a: np.ndarray
      The array which has values that are to be replaced.
    mask: np.ndarray of bools
      An array of booleans whose True values denote which of array 'a's values are to be replaced.
    replace_with: np.ndarray
      The values to be used to replace the corresponding values in 'a'.

    Returns
    -------
    dict(
      target: np.ndarray of same type as 'a'
        The array with the necessary values replaced.
      mask: np.ndarray of bools
        An array of booleans whose True values denote which of array 'a's values are to be replaced.
      replaced_vals: np.ndarray of same type as 'a'
        The values that were overwritten when they were replaced by the replace_with values.
      replace_with_shape: list of ints
        The original shape of the replace_with array.
    )

    """
    self.mask = mask
    # Cast the replace_with values to an array.
    replace_with = np.array(replace_with)
    target = ut.maybe_copy(a)

    # Save the values that are going to be replaced.
    replaced_vals = af.empty_array_like(a)

    replaced_vals[mask] = target[mask]

    # if len(replace_with.shape) != 1:
    #   raise ValueError("replace_with must be numpy array of rank 1, Got {} ".format(replace_with.shape))
    # if int(np.sum(mask)) != int(replace_with.size):
    #   raise ValueError("Number of values to be replaced needs to match the size of replace_with. Got: {} and {}".format(np.sum(mask), replace_with.size))

    # Replace the values with the values found in replace_with.
    target[mask] = replace_with

    return {'target': target, 'mask': mask, 'replaced_vals': replaced_vals, 'replace_with': replace_with}

  def _pump(self, target, mask, replaced_vals, replace_with):
    """Execute the Replace tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray of same type as 'a'
      The array with the necessary values replaced.
    mask: np.ndarray of bools
      An array of booleans whose True values denote which of array 'a's values are to be replaced.
    replaced_vals: np.ndarray of same type as 'a'
      The values that were overwritten when they were replaced by the replace_with values.
    replace_with_shape: list of ints
      The original shape of the replace_with array.

    Returns
    -------
    dict(
      a: np.ndarray
        The array which has values that are to be replaced.
      mask: np.ndarray of bools
        An array of booleans whose True values denote which of array 'a's values are to be replaced.
      replace_with: np.ndarray
        The values to be used to replace the corresponding values in 'a'.
    )

    """
    a = ut.maybe_copy(target)
    replaced_vals = np.array(replaced_vals)
    if replaced_vals.dtype.itemsize > a.dtype.itemsize:
      a = a.astype(replaced_vals.dtype)

    if replaced_vals.size == 1:
      a[mask] = replaced_vals
    else:
      a[mask] = replaced_vals[mask]
    a = a.astype(replaced_vals.dtype.type)
    return {'a': a, 'mask': mask, 'replace_with': replace_with}
