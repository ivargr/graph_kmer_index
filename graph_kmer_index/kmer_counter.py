import logging
import time


from npstructures import Counter, HashTable
import numpy as np


def choose_modulo(n_elements):
    if n_elements < 1000000:
        return 2000003
    elif n_elements < 10000000:
        return 19999999
    else:
        return 200000003



class KmerCounter:
    def __init__(self, counter):
        self.counter = counter

    @classmethod
    def from_flat_kmersv2(cls, flat, modulo, subsample_ratio=1):
        logging.info("Subsampling ration: %d" % subsample_ratio)
        kmers = flat._hashes
        logging.info("Kmers beforesubsampling: %d" % len(kmers))
        kmers = kmers[::subsample_ratio]
        logging.info("Kmers after subsampling: %d" % len(kmers))

        return cls.from_kmers(kmers, modulo)

    @classmethod
    def from_kmers(cls, kmers, modulo):
        logging.info("Finding unique kmers")
        t = time.perf_counter()
        unique_kmers, counts = np.unique(kmers, return_counts=True)
        logging.info("DOne finding unique kmers")
        if modulo == 0:
            modulo = choose_modulo(len(unique_kmers))
            logging.info("Choosing suitable modulo for hashtable to be %d" % modulo)
        counter = HashTable(unique_kmers, counts, mod=modulo)
        return cls(counter)

    @classmethod
    def from_flat_kmers(cls, flat, modulo, chunk_size=50000000):
        kmers = flat._hashes
        logging.info("Finding unique kmers")
        t = time.perf_counter()
        unique_kmers = np.unique(kmers)
        logging.info("Time spent to get unique kmers from %d kmers: %d sec" % (len(kmers), time.perf_counter()-t))
        logging.info("Unique kmers: %d" % len(unique_kmers))

        np.save("debugging", unique_kmers)
        logging.info("Saved debugging file")

        logging.info("Making counter")
        t = time.perf_counter()
        counter = Counter(unique_kmers, mod=modulo)
        logging.info("Time spent making counter: %d" % (time.perf_counter()-t))

        logging.info("Counting")
        t = time.perf_counter()
        for i, chunk in enumerate(np.array_split(kmers, 1 + len(kmers) // chunk_size)):
            logging.info("Counting chunk %d" % i)
            counter.count(chunk)

        #counter.count(kmers)
        logging.info("Time spent counting: %d" % (time.perf_counter()-t))
        return cls(counter)

    def get_frequency(self, kmer):
        #assert isinstance(kmer, int), "Kmer is %s" % type(kmer)
        return self.counter[int(kmer)]

    def score_kmers(self, kmers):
        # to be used as a scorer for a set of kmers
        hits = [self.counter[int(k)] for k in kmers]
        hits = [h[0] for h in hits if len(h) > 0]
        # low score is bad
        if len(hits) == 0:
            return 1
        return -np.max(hits)
