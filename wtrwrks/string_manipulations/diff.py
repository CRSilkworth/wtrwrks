"""Functions which deal with differences between strings."""
import difflib
import json


def get_diff_string(source_string, target_string):
  """Generate a string which can be used to transform one string into another.

  Parameters
  ----------
  source_string : str
    The string that uses the diff_string to get transformed into the target string.
  target_string : type
    The string that source string is being transformed into.

  Returns
  -------
  str
    A string that describes a sequence of operations on source_string to produce target_string.

  """
  # Split up the strings into lists.
  source_string = list(source_string)
  target_string = list(target_string)

  # Use a sequence matcher to identify all the common substrings of
  # source_string an target_string.
  matcher = difflib.SequenceMatcher(None, source_string, target_string)

  tag_map = {
    'delete': 'd',
    'insert': 'i',
    'replace': 'i'
  }

  # Get out the operation codes which list out all the operations that need to
  # occur in order to transform source_string into target_string.
  diff_string = []
  for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    if tag == 'equal':
      continue

    substring = ''
    if tag in ('insert', 'replace'):
      substring = ''.join(target_string[j1:j2])

    diff_string.append((tag_map[tag], i1, i2, substring))

  return json.dumps(diff_string)


def reconstruct(source_string, diff_string):
  """Generate a string which can be used to transform one string into another.

  Parameters
  ----------
  source_string : str
    The string that uses the diff_string to get transformed into the target string.
  target_string : str
    A string that describes a sequence of operations on source_string to produce target_string.

  Returns
  -------
  str
    The string that source string is transformed into using the diff string.

  """
  target_string = list(source_string)
  for tag, i1, i2, substring in reversed(json.loads(diff_string)):
    if tag == 'd':
      del target_string[i1:i2]
    elif tag == 'i':
      target_string[i1:i2] = substring

  return ''.join(target_string)
