import reversible_transforms.rev_array.rev_array as ra


class Operation(object):
  def __init__(self, op_name, graph):
    self.op_name = op_name
    self.op_cons = {
      'forward_inputs': {},
      'forward_outputs': {},
      'backward_inputs': {},
      'backward_outputs': {}
    }

    if self in graph.operations:
      raise ValueError(self.op_name + " already name of op in graph.")

    graph.operations.add(self)

  def _setup_inputs(self, inputs, graph):
    for kw in inputs:
      if inputs[kw] is None:
          self.op_cons['forward_inputs'][kw] = self.op_cons['forward_outputs'][kw]
          self.op_cons['backward_outputs'][kw] = self.op_cons['forward_outputs'][kw]
          continue
      # Forward inputs
      forward_input = inputs[kw]
      forward_input.in_keys.append((self, kw))

      self.op_cons['forward_inputs'][kw] = forward_input

      if forward_input in graph.forward_outputs:
        graph.forward_outputs.remove(forward_input)

      # Backward outputs
      backward_output = ra.OpCon(self, kw)
      self.op_cons['backward_outputs'][kw] = backward_output

      # Backward inputs
      alt_kw = forward_input.key
      backward_output.in_keys.append((forward_input.op, alt_kw))
      forward_input.op.op_cons['backward_inputs'][alt_kw] = backward_output

  def _setup_outputs(self, outputs, graph):
    for kw in outputs:
      # Forward outputs
      forward_output = ra.OpCon(self, kw)
      self.op_cons['forward_outputs'][kw] = forward_output
      self.op_cons['backward_inputs'][kw] = forward_output

      graph.op_cons['forward'].add(self[kw])
      graph.forward_outputs.add(self[kw])

  def forward(self, **inputs):
    pass

  def __getitem__(self, output_key):
    return self.op_cons['forward_outputs'][output_key]

  def __hash__(self):
    return hash(self.op_name)

  def __str__(self):
    return str(self.op_name)

class Placeholder(Operation):
  def __init__(self, op_name, graph):
    super(self.__class__, self).__init__(op_name=op_name, graph=graph)

    self._setup_outputs(['data'], graph)
    self._setup_inputs({'data': None}, graph)

    graph.forward_inputs.add(self['data'])

  def forward(self, data):
    return {'data': data}

  def backward(self, data):
    return {'data': data}


class Add(Operation):
  def __init__(self, op_name, graph, a, b):
    super(self.__class__, self).__init__(op_name=op_name, graph=graph)

    self._setup_inputs({'a': a, 'b': b}, graph)
    self._setup_outputs(['a', 'data'], graph)

  def forward(self, a, b):
    return {'data': a + b, 'a': a}

  def backward(self, a, data):
    return {'a': a, 'b': data - a}
