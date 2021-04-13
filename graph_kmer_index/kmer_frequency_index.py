import numpy as np
import logging


class KmerFrequencyIndex:
    def __init__(self, kmers, frequencies):
        self._kmers = kmers
        self._frequencies = frequencies

    def get(self, kmer):
        index = np.searchsorted(self._kmers, kmer, side="right")
        if self._kmers[index] == kmer:
            return self._frequencies[index]

        logging.warning("No hit for kmer %d" % kmer)
        return 0

    @classmethod
    def from_kmers(cls, kmers):
        logging.info("Sorting")
        sorting = np.argsort(kmers)
        kmers = kmers[sorting]
        logging.info("Counting")
        unique, frequencies = np.unique(kmers, return_counts=True)
        return cls(unique, frequencies)

    def to_file(self, file_name):
        np.savez(file_name, kmers=self._kmers, frequencies=self._frequencies)

    @classmethod
    def from_file(cls, file_name):
        try:
            data = np.load(file_name)
        except FileNotFoundError:
            data = np.load(file_name + ".npz")

        return cls(data["kmers"], data["frequencies"])