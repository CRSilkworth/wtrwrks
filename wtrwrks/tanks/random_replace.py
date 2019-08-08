"""Replace tank definition."""
import wtrwrks.waterworks.waterwork_part as wp
import wtrwrks.waterworks.tank as ta
import wtrwrks.tanks.utils as ut
import wtrwrks.utils.array_functions as af
import numpy as np


class RandomReplace(ta.Tank):
  """The defintion of the Replace tank. Contains the implementations of the _pour and _pump methods, as well as which slots and tubes the waterwork objects will look for.

  Attributes
  ----------
  slot_keys: list of strs
    The names off all the tank's slots, i.e. inputs in the pour (forward) direction, or outputs in the pump (backward) direction
  tubes: list of strs
    The names off all the tank's tubes, i.e. outputs in the pour (forward) direction,

  """

  func_name = 'random_replace'
  slot_keys = ['a', 'replace_with', 'do_not_replace_vals', 'prob', 'max_replace']
  tube_keys = ['target', 'replaced_vals', 'replace_with', 'do_not_replace_vals', 'prob', 'mask_mask', 'mask_positions', 'max_replace']
  pass_through_keys = ['replace_with', 'do_not_replace_vals', 'prob', 'max_replace']

  def _pour(self, a, replace_with, prob, do_not_replace_vals, max_replace):
    """Randomly replace the elements of a with values 'replace_with'. The variable 'prob' determines the probability any single value will be replaced except for those values which are in 'do_not_replace_vals'. Therefore if a large portion of the elements fall into the do_not_replace_vals, or they run into the max_replace, the percentage of 'a's values that are replaced will be significantly below 'prob'.

    Parameters
    ----------
    a: np.ndarray
      The array which has values that are to be replaced.
    replace_with: np.ndarray
      The value to be used to replace the corresponding values in 'a'.
    prob: 0 <= float <= 1
      The probability that each value is replaced
    max_replace: int <= a.shape[-1]
      The maximum allowed replacements along the last dimension.
    do_not_replace_vals: np.ndarray
      Values to skip when randomly replacing.
    max_replace: int
      The maximum number of allowed replacements in the last dimension.

    Returns
    -------
    dict(
      target: np.ndarray of same type as 'a'
        The array with the necessary values replaced.
      mask_mask: np.ndarray of bools
        An array of booleans whose True values denote which of array 'a's values were replaced.
      mask_positions: np.ndarray of bools
        The positions of the masked values
      prob: 0 <= float <= 1
        The probability that each value is replaced
      replaced_vals: np.ndarray of same type as 'a'
        The values that were overwritten when they were replaced by the replace_with values.
      do_not_replace_vals: np.ndarray
        Values to skip when randomly replacing.
      max_replace: int <= a.shape[-1]
        The maximum allowed replacements along the last dimension.
    )

    """
    self.a = a
    # Cast the replace_with values to an array.
    replace_with = np.array(replace_with)
    target = ut.maybe_copy(a)

    a_mask = np.reshape(np.random.choice([True, False], size=a.size, p=[prob, 1.0 - prob]), a.shape)
    a_mask = a_mask & ~np.isin(a, do_not_replace_vals)

    mask_positions = np.argsort(a_mask, axis=-1)
    mask_positions = np.flip(mask_positions, axis=-1)

    mask_mask = np.sort(a_mask, axis=-1)
    mask_mask = np.flip(mask_mask, axis=-1)
    mask_mask[..., max_replace:] = False

    mask_positions[~mask_mask] = -1

    indices = list(np.unravel_index(np.arange(a.size, dtype=int), a.shape))
    indices[-1] = mask_positions.flatten()

    replaced_vals = target[indices]
    replaced_vals = np.reshape(replaced_vals, target.shape)

    indices_mask = ~(mask_positions.flatten() == -1)
    trunc_indices = []
    for dim in indices:
      trunc_indices.append(dim[indices_mask])

    target[trunc_indices] = replace_with

    replaced_vals = replaced_vals[..., :max_replace]
    mask_positions = mask_positions[..., :max_replace]
    mask_mask = mask_mask[..., :max_replace]

    return {'target': target, 'replaced_vals': replaced_vals, 'do_not_replace_vals': do_not_replace_vals, 'prob': prob, 'mask_mask': mask_mask, 'mask_positions': mask_positions, 'replace_with': replace_with}

  def _pump(self, target, mask_mask, mask_positions, replace_with, replaced_vals, do_not_replace_vals, prob, max_replace):
    """Execute the Replace tank (operation) in the pump (backward) direction.

    Parameters
    ----------
    target: np.ndarray of same type as 'a'
      The array with the necessary values replaced.
    mask_mask: np.ndarray of bools
      An array of booleans whose True values denote which of array 'a's values were replaced.
    mask_positions: np.ndarray of bools
      The positions of the masked values
    prob: 0 <= float <= 1
      The probability that each value is replaced
    replaced_vals: np.ndarray of same type as 'a'
      The values that were overwritten when they were replaced by the replace_with values.
    do_not_replace_vals: np.ndarray
      Values to skip when randomly replacing.
    max_replace: int <= a.shape[-1]
      The maximum allowed replacements along the last dimension.

    Returns
    -------
    dict(
      a: np.ndarray
        The array which has values that are to be replaced.
      replace_with: np.ndarray
        The value to be used to replace the corresponding values in 'a'.
      prob: 0 <= float <= 1
        The probability that each value is replaced
      max_replace: int <= a.shape[-1]
        The maximum allowed replacements along the last dimension.
      do_not_replace_vals: np.ndarray
        Values to skip when randomly replacing.
      max_replace: int
        The maximum number of allowed replacements in the last dimension.
    )

    """

    a = ut.maybe_copy(target)
    if replaced_vals.dtype.itemsize > a.dtype.itemsize:
      a = a.astype(replaced_vals.dtype)

    pad_num = max(0, a.shape[-1] - max_replace)
    mask_positions = self._pad_last_dim(mask_positions, -1, pad_num)
    mask_mask = self._pad_last_dim(mask_mask, False, pad_num)
    replaced_vals = self._pad_last_dim(replaced_vals, u'', pad_num)

    indices = list(np.unravel_index(np.arange(a.size, dtype=int), a.shape))
    indices[-1] = mask_positions.flatten()
    indices_mask = ~(mask_positions.flatten() == -1)
    #
    trunc_indices = []
    for dim in indices:
      trunc_indices.append(dim[indices_mask])
    a[trunc_indices] = replaced_vals[mask_mask]
    # a[trunc_indices] = replaced_vals[mask_mask]

    a = a.astype(replaced_vals.dtype.type)

    return {'a': a, 'replace_with': replace_with, 'do_not_replace_vals': do_not_replace_vals, 'prob': prob}

  def _pad_last_dim(self, a, val, num):
    shape = list(a.shape[:-1]) + [num]
    empties = np.full(shape, val, a.dtype)
    return np.concatenate([a, empties], axis=-1)
