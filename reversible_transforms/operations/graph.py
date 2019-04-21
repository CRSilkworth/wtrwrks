import reversible_transforms.operations.operation as o
import numpy as np
import pprint


class Graph(object):
    def __init__(self):
      self.operations = set()
      self.forward_inputs = set()
      self.forward_outputs = set()
      self.op_cons = {
        'forward': set(),
        'backward': set()
      }

    def __enter__(self):
      return self

    def __exit__(self, exc_type, exc_val, exc_tb):
      pass

    def forward(self, feed_dict):
      input_keys = set(feed_dict.keys())
      ph_keys = set([i for i in self.forward_inputs])

      if input_keys != ph_keys:
        raise ValueError("Must input one keyword argument for every placeholder. Got " + str(list(input_keys)) + " for inputs " + str(list(ph_keys)))

      op_order = self.get_forward_op_order()

      calc = {}
      for ph in feed_dict:
        calc[ph.op] = ph.op.forward(feed_dict[ph])

      for op in op_order:
        if op in calc:
          continue
        inputs = {}

        for kw in op.op_cons['forward_inputs']:
          op_con = op.op_cons['forward_inputs'][kw]
          inputs[kw] = calc[op_con.op][op_con.key]

        calc[op] = op.forward(**inputs)
      outputs = {}

      for op_con in self.forward_outputs:
        outputs[op_con] = calc[op_con.op][op_con.key]

      return outputs

    def get_forward_op_order(self):
      op_cons = []
      op_order = []
      accounted_for = set()
      for op_con in self.forward_outputs:
        if op_con.op in accounted_for:
          continue
        op_cons.append(op_con)
        op_order.insert(0, op_con.op)
        accounted_for.add(op_con.op)

      while op_cons:
        op_con = op_cons.pop()
        op = op_con.op

        if type(op) is not o.Placeholder:
          for nk in op.op_cons['forward_inputs']:
            dep_con = op.op_cons['forward_inputs'][nk]
            if dep_con.op in accounted_for:
              continue
            op_cons.append(dep_con)
            op_order.insert(0, dep_con.op)
            accounted_for.add(dep_con.op)

      return op_order

    def backward(self, feed_dict):
      input_keys = set(feed_dict.keys())
      output_keys = set([i for i in self.forward_outputs])

      if input_keys != output_keys:
        raise ValueError("Must input one keyword argument for every placeholder. Got " + str([str(i) for i in input_keys]) + " for inputs " + str([str(i) for i in output_keys]))

      op_order = self.get_backward_op_order()

      calc = {}
      calc.update(feed_dict)

      # for k, v in calc.iteritems():
        # print k, v

      for op in op_order:
        inputs = {}
        for kw in op.op_cons['backward_inputs']:
          op_con = op.op_cons['backward_inputs'][kw]
          inputs[kw] = calc[op_con]

        backward_outputs = op.backward(**inputs)
        for kw in op.op_cons['backward_outputs']:
          op_con = op.op_cons['backward_outputs'][kw]
          calc[op_con] = backward_outputs[kw]

      inputs = {}
      for op_con in self.forward_inputs:
        inputs[op_con] = calc[op_con]

      return inputs

    def get_backward_op_order(self):
      op_cons = []
      op_order = []
      accounted_for = set()

      for op_con in self.forward_inputs:
        if op_con.op in accounted_for:
          continue
        op_cons.append(op_con)
        op_order.insert(0, op_con.op)
        accounted_for.add(op_con.op)

      while op_cons:
        op_con = op_cons.pop()
        op = op_con.op

        for nk in op.op_cons['backward_inputs']:
          dep_con = op.op_cons['backward_inputs'][nk]
          if dep_con is None or dep_con.op in accounted_for:
            continue
          op_cons.append(dep_con)
          op_order.insert(0, dep_con.op)
          accounted_for.add(dep_con.op)

      return op_order

if __name__ == '__main__':
  with Graph() as graph:
    ph1 = o.Placeholder('ph1', graph)
    ph2 = o.Placeholder('ph2', graph)

    add1 = o.Add('add1', graph, ph1['data'], ph2['data'])
    add2 = o.Add('add2', graph, add1['data'], ph1['data'])

  # print [str(op_con) for op_con in graph.get_forward_op_order()]

  feed_dict = {ph1['data']: np.array([1, 2]), ph2['data']: np.array([3, 4])}
  outputs = graph.forward(feed_dict)
  back_feed_dict = outputs

  # print [str(op_con) for op_con in graph.get_backward_op_order()]

  inputs = graph.backward(outputs)
  for input in inputs:
    print input, inputs[input]
