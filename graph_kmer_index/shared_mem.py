import random
import pickle
import time

import SharedArray as sa
import logging
import numpy as np
from itertools import repeat
from pathos.multiprocessing import Pool

class SingleSharedArray:
    properties = {"array"}
    def __init__(self, array=None):
        self.array = array

SHARED_MEMORIES_IN_SESSION = []

_shared_pool = None


def get_shared_pool(n_threads=16):
    global _shared_pool
    if _shared_pool is None:
        _shared_pool = Pool(n_threads)
        logging.info("Made shared pool")

    return _shared_pool


def close_shared_pool():
    global _shared_pool
    if _shared_pool is not None:
        _shared_pool.close()
        logging.info("Closed shared pool")


def to_shared_memory(object, name):
    global SHARED_MEMORIES_IN_SESSION
    t = time.perf_counter()
    logging.debug("Writing to shared memory %s" % name)
    meta_information = {}
    for property_name in object.properties:
        data = object.__getattribute__(property_name)

        if data is None:
            data = np.zeros(0)

        # Wrap single ints in arrays
        if data.shape == ():
            data = np.array([data], dtype=data.dtype)

        data_type = data.dtype
        data_shape = data.shape
        meta_information[property_name] = (data_type, data_shape)

        # Make shared memory and copy data to buffer
        #logging.info("Field %s has shape %s and type %s" % (property_name, data_shape, data_type))
        array_name = name + "__" + property_name
        try:
            sa.delete(array_name)
        except FileNotFoundError:
            logging.debug("No existing shared memory, can create new one")

        shared_array = sa.create(array_name, data_shape, data_type)
        shared_array[:] = data
        SHARED_MEMORIES_IN_SESSION.append(array_name)

    f = open(name + "_meta.shm", "wb")
    pickle.dump(meta_information, f)
    #logging.info("Done writing to shared memory. Took %.3f sec" % (time.perf_counter()-t))
    #logging.info("Wrote meta data to file")


def from_shared_memory(cls, name):
    object = cls()
    t = time.perf_counter()
    try:
        meta_data = pickle.load(open(name + "_meta.shm", "rb"))
    except EOFError:
        logging.error("Failed reading from %s._meta.shm" % name)
        raise

    for property_name, data in meta_data.items():
        data_type = data[0]
        data_shape = data[1]
        #logging.info("Found property %s with shape %s and type %s" % (property_name, data_shape, data_type))
        data = sa.attach(name + "__" + property_name)
        # Single ints are wrapped in arrays
        if len(data) == 1 and property_name == "_modulo":
            data = data[0]
            #logging.info("Extracted single int from %s" % property_name)
        setattr(object, property_name, data)
        #print("Data sample from class:: %s" % object.__getattribute__(property_name))


    return object


def remove_shared_memory(name):
    shared_memories = [s.name.decode("utf-8") for s in sa.list()]

    for m in shared_memories:
        if m.startswith(name + "__"):
            sa.delete(m)
            return

    logging.warning("No shared memory with name %s" % name)
    logging.warning("Available shared memories: %s" % shared_memories)


def remove_shared_memory_in_session():
    for name in SHARED_MEMORIES_IN_SESSION:
        try:
            sa.delete(name)
        except FileNotFoundError:
            logging.warning("Tried to deleted shared memory %s that did not exist" % name)

def remove_all_shared_memory():
    for shared in sa.list():
        sa.delete(shared.name.decode("utf-8"))


def _run_numpy_based_function_on_shared_memory_arguments(function, arguments, interval):
    start_time = time.perf_counter()
    start, end = interval
    #logging.info("Running on interval %d-%d" % (start, end))
    arguments = [from_shared_memory(SingleSharedArray, a).array if type(a) == str else a for a in arguments]
    sliced_arguments = []
    for argument in arguments:
        if isinstance(argument, np.ndarray):
            argument = argument[start:end]
        sliced_arguments.append(argument)

    result = function(*sliced_arguments)
    shared_memory_name = str(random.randint(0,10e15))
    to_shared_memory(SingleSharedArray(result), shared_memory_name)
    #logging.info("Interval %d-%d took %.4f sec" % (start, end, time.perf_counter()-start_time))
    return shared_memory_name


def run_numpy_based_function_in_parallel(function, n_threads, arguments):
    # split every argument that is not an int
    # put each in shared memory
    # run function on chunk on separete thread, put in result array
    # return result array
    new_arguments = []

    # Put np arrays in shared memory, everything else we keep as is
    array_length = 0
    for argument in arguments:
        if isinstance(argument, np.ndarray):
            argument_id = str(np.random.randint(0, 10e15))
            to_shared_memory(SingleSharedArray(argument), argument_id)
            new_arguments.append(argument_id)
            assert array_length == 0 or array_length == argument.shape[0], "Found argument with different shape"
            array_length = argument.shape[0]

        else:
            new_arguments.append(argument)

    t = time.perf_counter()
    pool = get_shared_pool(n_threads)  # Pool(n_threads)

    intervals = list([int(i) for i in np.linspace(0, array_length, n_threads + 1)])
    intervals = [(from_pos, to_pos) for from_pos, to_pos in zip(intervals[0:-1], intervals[1:])]
    #logging.info("Will run on intervals %s" % intervals)

    results = []
    for result in pool.starmap(_run_numpy_based_function_on_shared_memory_arguments, zip(repeat(function), repeat(new_arguments), intervals)):
        result = from_shared_memory(SingleSharedArray, result).array
        results.append(result)

    t = time.perf_counter()
    results = np.concatenate(results)
    logging.info("Time to concatenate results: %.3f" % (time.perf_counter()-t))
    return results




