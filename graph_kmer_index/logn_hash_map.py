import numpy as np
import logging

class BaseHashMap:
    def to_file(self, file_name):
        np.save(file_name, self._hashes)

    @classmethod
    def from_file(cls, file_name):
        data = np.load(file_name + ".npy")
        if data.dtype != np.int:
            data = data.astype(np.int)
        #map = cls([])
        #map._hashes = data
        return cls(data)

    def unhash(self, hash):
        return self._hashes[hash]


class ModuloHashMap(BaseHashMap):
    def __init__(self, hashes):
        if hashes.dtype != np.int:
            logging.info("Converting hashes to int")
            self._hashes = hashes.astype(np.int) # np.array(hashes).astype(np.int)
        else:
            self._hashes = hashes

    @classmethod
    def from_sorted_array(cls, sorted_hash_array, modulo=452930477):
        logging.info("Creating modulohashmap with modulo %s" % modulo)
        sorted_hash_array = np.unique(sorted_hash_array)
        hashes = np.zeros(modulo)
        modulo = sorted_hash_array % modulo
        hashes[modulo] = np.array(np.arange(0, len(sorted_hash_array)), dtype=np.uint32)
        logging.info("Done creating hashmap")
        return cls(hashes)

    def hash(self, key, modulo=452930477):
        #return int(self._hashes[key])
        index = self._hashes[key % modulo]
        if index == 0:
            return None

        return int(index)


class LogNHashMap(BaseHashMap):
    def __init__(self, sorted_hash_array):
        self._hashes = np.unique(sorted_hash_array)

    def hash(self, key):
        index = np.searchsorted(self._hashes, key)
        if index >= len(self._hashes) or self._hashes[index] != key:
            return None
        return index


