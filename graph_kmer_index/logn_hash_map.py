import numpy as np

class LogNHashMap:
    def __init__(self, sorted_hash_array):
        self._hashes = np.unique(sorted_hash_array)

    def hash(self, key):
        index = np.searchsorted(self._hashes, key)
        if self._hashes[index] != key:
            return None
        return index

    def to_file(self, file_name):
        np.save(file_name, self._hashes)

    @classmethod
    def from_file(cls, file_name):
        data = np.load(file_name + ".npy")
        map = cls([])
        map._hashes = data
        return map

    def unhash(self, hash):
        return self._hashes[hash]


