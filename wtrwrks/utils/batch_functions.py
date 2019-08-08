
def batcher(iterable, batch_size=None):
  if type(iterable) in (list, tuple):
    iterable = (i for i in iterable)

  out_batch = []
  break_out = False

  while True:
    try:
      out_batch.append(next(iterable))
    except StopIteration:
      break_out = True

    if len(out_batch) == batch_size or (break_out and out_batch):
      yield out_batch
      out_batch = []

    if break_out:
      break
