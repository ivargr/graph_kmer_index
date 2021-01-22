import pickle
import SharedArray as sa
import logging
import numpy as np

def to_shared_memory(object, name):
    logging.info("Writing to shared memory %s" % name)
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
        #logging.info("Field %s has shape %s and type %s" % (property_name, data_shape, data_type))
        try:
            sa.delete(name + "_" + property_name)
            logging.info("Deleted already shared memory")
        except FileNotFoundError:
            logging.info("No existing shared memory, can create new one")

        shared_array = sa.create(name + "_" + property_name, data_shape, data_type)
        shared_array[:] = data

    f = open(name + "_meta.shm", "wb")
    pickle.dump(meta_information, f)
    logging.info("Done writing to shared memory")
    #logging.info("Wrote meta data to file")


def from_shared_memory(cls, name):
    object = cls()
    meta_data = pickle.load(open(name + "_meta.shm", "rb"))
    for property_name, data in meta_data.items():
        data_type = data[0]
        data_shape = data[1]
        #logging.info("Found property %s with shape %s and type %s" % (property_name, data_shape, data_type))
        data = sa.attach(name + "_" + property_name)
        # Single ints are wrapped in arrays
        if len(data) == 1 and property_name == "_modulo":
            data = data[0]
            logging.info("Extracted single int from %s" % property_name)
        setattr(object, property_name, data)
        #print("Data sample from class:: %s" % object.__getattribute__(property_name))

    return object


