import reversible_transforms.rev_array.rev_array as ra
import reversible_transforms.operations.operation as o


def _add_forward(a, b, op_name):
  r_dict = {
    'data': ra.OpNode(a + b, op_name, 'data'),
    'a': ra.OpNode(a, op_name, 'a')
  }
  return r_dict


def _add_backward(a, data, op_name):
  r_dict = {
    'b': ra.OpNode(data - a, op_name, 'b'),
    'a': ra.OpNode(a, op_name, 'a')
  }
  return r_dict


add = o.Operation(
  input_keys=['a', 'b', 'name'],
  output_keys=['data', 'a'],
  forward=_add_forward,
  backward=_add_backward
)


def _subtract_forward(a, b, op_name):
  r_dict = {
    'data': ra.OpNode(a - b, op_name, 'data'),
    'a': ra.OpNode(a, op_name, 'a')
  }
  return r_dict


def _subtract_backward(a, data, op_name):
  r_dict = {
    'b': ra.OpNode(a - data, op_name, 'b'),
    'a': ra.OpNode(a, op_name, 'a')
  }
  return r_dict


subtract = o.Operation(
  input_keys=['a', 'b', 'name'],
  output_keys=['data', 'a'],
  forward=_subtract_forward,
  backward=_subtract_backward
)
