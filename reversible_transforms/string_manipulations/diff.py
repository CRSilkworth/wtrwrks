import difflib
import json


def get_diff_string(source_string, target_string):
  source_string = list(source_string)
  target_string = list(target_string)
  matcher = difflib.SequenceMatcher(None, source_string, target_string)

  tag_map = {
    'delete': 'd',
    'insert': 'i',
    'replace': 'i'
  }
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
  target_string = list(source_string)
  for tag, i1, i2, substring in reversed(json.loads(diff_string)):
    if tag == 'd':
      del target_string[i1:i2]
    elif tag == 'i':
      target_string[i1:i2] = substring

  return ''.join(target_string)
