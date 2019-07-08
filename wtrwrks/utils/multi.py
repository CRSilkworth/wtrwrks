import pathos.multiprocessing as mp


def multi_map(func, iterable, num_threads=1, use_threading=False):
  out_list = []
  if num_threads != 1:
    if not use_threading:
      # pool = mp.ProcessPool(num_threads)
      pool = mp.ProcessingPool(num_threads)
    else:
      pool = mp.ThreadPool(num_threads)

  # If multithreading run pool.map, otherwise just run recon_and_pour
  if num_threads != 1:
    out_list = pool.map(func, iterable)

    # processingpool carries over information. Need to terminate a restart
    # to prevent memory leaks.
    pool.terminate()
    if not use_threading:
      pool.restart()
  else:
    for element in iterable:
      out_list.append(func(element))
  return out_list
