import logging
import numpy as np


def kmer_hash_to_reverse_complement_hash(hash, k):
    return kmer_hashes_to_reverse_complement_hash(np.array([hash]), k)[0]


def kmer_hashes_to_reverse_complement_hash(hashes, k):
    assert k <= 31
    reverse_power_array = np.power(4, np.arange(0, k, dtype=np.uint64), dtype=np.uint64)

    hashes = hashes.astype(np.uint64)
    bases = np.zeros((len(hashes), k), dtype=np.uint64)
    for i in range(k):
        logging.info("Finding hashes. k=%d/%d" % (i, k))
        # print("Hash now: %d" % hash)
        exponential = np.power(np.uint64(4), np.uint64(k - i - 1), dtype=np.uint64)
        # base = hash // exponential   # gives float, no good
        base = np.floor_divide(hashes, exponential, dtype=np.uint64)
        hashes -= base * exponential

        bases[:,i] = base

    complement_bases = (bases+2) % 4
    reverse_complement_hash = np.sum(complement_bases * reverse_power_array, axis=1)
    return reverse_complement_hash