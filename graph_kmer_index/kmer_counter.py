import logging
import time

from npstructures import Counter
import numpy as np


class KmerCounter:
    def __init__(self, counter):
        self.counter = counter

    @classmethod
    def from_flat_kmers(cls, flat, modulo):
        kmers = flat._hashes
        logging.info("Finding unique kmers")
        t = time.perf_counter()
        unique_kmers = np.unique(kmers)
        logging.info("Time spent to get unique kmers: %d sec" % (time.perf_counter()-t))

        logging.info("Making counter")
        t = time.perf_counter()
        counter = Counter(unique_kmers, mod=modulo)
        logging.info("Time spent making counter: %d" % (time.perf_counter()-t))

        logging.info("Counting")
        t = time.perf_counter()
        counter.count(kmers)
        logging.info("Time spent counting: %d" % (time.perf_counter()-t))
        return cls(counter)

    def get_frequency(self, kmer):
        return self.counter[kmer]

