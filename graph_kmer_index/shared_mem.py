import pickle
import SharedArray as sa
import logging
import numpy as np

class SingleSharedArray:
    properties = {"array"}
    def __init__(self, array=None):
        self.array = array

SHARED_MEMORIES_IN_SESSION = []

def to_shared_memory(object, name):
    global SHARED_MEMORIES_IN_SESSION
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
    logging.debug("Done writing to shared memory")
    #logging.info("Wrote meta data to file")


def from_shared_memory(cls, name):
    object = cls()
    meta_data = pickle.load(open(name + "_meta.shm", "rb"))
    for property_name, data in meta_data.items():
        data_type = data[0]
        data_shape = data[1]
        #logging.info("Found property %s with shape %s and type %s" % (property_name, data_shape, data_type))
        data = sa.attach(name + "__" + property_name)
        # Single ints are wrapped in arrays
        if len(data) == 1 and property_name == "_modulo":
            data = data[0]
            logging.info("Extracted single int from %s" % property_name)
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
    logging.warning("Available shared memories: %s" % available)


def remove_shared_memory_in_session():
    for name in SHARED_MEMORIES_IN_SESSION:
        try:
            sa.delete(name)
            logging.info("Deleting shared memory %s" % name)
        except FileNotFoundError:
            logging.warning("Tried to deleted shared memory that did not exist")

def remove_all_shared_memory():
    for shared in sa.list():
        logging.info("Deleting %s" % shared.name.decode("utf-8"))
        sa.delete(shared.name.decode("utf-8"))
