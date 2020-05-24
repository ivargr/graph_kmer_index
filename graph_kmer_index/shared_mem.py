import pickle
from multiprocessing import shared_memory
import logging
import numpy as np

def to_shared_memory(object, name):
    meta_information = {}
    for property_name in object.properties:
        data = object.__getattribute__(property_name)
        # Wrap single ints in arrays
        if data.shape == ():
            data = np.array([data], dtype=data.dtype)

        data_type = data.dtype
        data_shape = data.shape
        meta_information[property_name] = (data_type, data_shape)

        # Make shared memory and copy data to buffer
        logging.info("Field %s has shape %s and type %s" % (property_name, data_shape, data_type))
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes, name=name + "_" + property_name)
        holder = np.ndarray(data_shape, dtype=data_type, buffer=shm.buf)
        holder[:] = data[:]
        shm.close()


    f = open(name + "_meta.shm", "wb")
    pickle.dump(meta_information, f)
    logging.info("Wrote meta data to file")


def from_shared_memory(cls, name):
    object = cls()
    meta_data = pickle.load(open(name + "_meta.shm", "rb"))
    for property_name, data in meta_data.items():
        data_type = data[0]
        data_shape = data[1]
        logging.info("Found property %s with shape %s and type %s" % (property_name, data_shape, data_type))
        shm = shared_memory.SharedMemory(name=name + "_" + property_name)
        data = np.ndarray(data_shape, dtype=data_type, buffer=shm.buf)
        print("Data sample: %d" % data[0])
        # Single ints are wrapped in arrays
        if len(data) == 1:
            data = data[0]
            logging.info("Extracted single int from %s" % property_name)
        setattr(object, property_name, data)
        print("Data sample from class:: %s" % object.__getattribute__(property_name))

    print("TEst .")
    print("Test", object.__getattribute__("_nodes"))
    return object


