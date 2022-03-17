import logging
import numpy as np


class CountMinSketchKmerCounter:
    def __init__(self, data, modulos):
        self._data = data
        self._modulos = modulos
        self._array_positions = np.concatenate([[0], np.cumsum(modulos)[:-1]])

    def _indexes(self, kmer):
        return (kmer % self._modulos) + self._array_positions

    def get_count(self, kmer):
        return np.min(self._data[self._indexes(kmer)])

    def get_counts(self, kmers):
        pass

    @classmethod
    def create_empty(cls, modulos, dtype=np.uint16):
        data = np.zeros(np.sum(modulos), dtype=dtype)
        return cls(data, np.asanyarray(modulos))

    def count_kmers(self, kmers):
        for mod, offset in zip(self._modulos, self._array_positions):
            indexes = (kmers % mod) + offset
            counts = np.bincount(indexes).astype(self._data.dtype)
            self._data[0:len(counts)] += counts
