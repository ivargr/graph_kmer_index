import logging
import numpy as np


def kmer_hash_to_reverse_complement_hash(hash, k):
    return kmer_hashes_to_reverse_complement_hash(np.array([hash]), k)[0]


def kmer_hashes_to_reverse_complement_hash(hashes, k):
    assert k <= 31
    reverse_power_array = np.power(4, np.arange(0, k, dtype=np.uint64), dtype=np.uint64)
    complement_bases = kmer_hashes_to_complement_bases(hashes, k)
    reverse_complement_hash = np.sum(complement_bases * reverse_power_array, axis=1)
    return reverse_complement_hash


def kmer_hashes_to_complement_hashes(hashes, k):
    assert k <= 31
    power_array = np.power(4, np.arange(0, k, dtype=np.uint64), dtype=np.uint64)[::-1]
    complement_bases = kmer_hashes_to_complement_bases(hashes, k)
    complement_hash = np.sum(complement_bases * power_array, axis=1)
    return complement_hash



def kmer_hashes_to_complement_bases(hashes, k):
    #raise NotImplementedError("Must be fixed to work with ACGT encoding")
    bases = kmer_hashes_to_bases(hashes, k)
    complement_bases = np.zeros_like(bases)
    complement_bases[bases == 0] = 3
    complement_bases[bases == 3] = 0
    complement_bases[bases == 2] = 1
    complement_bases[bases == 1] = 2
    #complement_bases = (bases + 2) % 4
    return complement_bases



def kmer_hashes_to_bases(hashes, k):
    hashes = hashes.astype(np.uint64)
    bases = np.zeros((len(hashes), k), dtype=np.uint64)
    for i in range(k):
        logging.debug("Finding hashes. k=%d/%d" % (i, k))
        # print("Hash now: %d" % hash)
        exponential = np.power(np.uint64(4), np.uint64(k - i - 1), dtype=np.uint64)
        # base = hash // exponential   # gives float, no good
        base = np.floor_divide(hashes, exponential, dtype=np.uint64)
        hashes -= base * exponential

        bases[:, i] = base
    return bases