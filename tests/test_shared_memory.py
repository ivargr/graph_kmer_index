import time
import logging

from shared_memory_wrapper.shared_memory import run_numpy_based_function_in_parallel
import numpy as np


def simple_numpy_function(a, b):
    return a + b

def slow_numpy_function(a, b, c, d):
    for i in range(10):
        b = b + a
        b = b-100
        c = 0 * (np.sqrt(c) * np.sqrt(c) * np.sqrt(c) * np.sqrt(c+a))
        e = (a + b - c)*d

    return e


def simple_test():
    a = np.arange(10) + 10
    b = np.zeros(10) + 20
    result = run_numpy_based_function_in_parallel(simple_numpy_function, 4, [a, b])
    linear_result = simple_numpy_function(a, b)
    assert np.all(result == linear_result)

def complex_test():
    size = 10000000
    a = np.arange(size) + 1.5
    b = np.zeros(size) + 1.5
    c = np.zeros(size) + 1.5
    d = 1.0
    start = time.perf_counter()
    result = run_numpy_based_function_in_parallel(slow_numpy_function, 10, [a, b, c, d])
    logging.info("Spent %.4f sec in parallel" % (time.perf_counter()-start))

    start = time.perf_counter()
    linear_result = slow_numpy_function(a, b, c, d)
    logging.info("Spent %.4f sec linear" % (time.perf_counter()-start))

    assert np.all(result == slow_numpy_function(a, b, c, d))


simple_test()
complex_test()