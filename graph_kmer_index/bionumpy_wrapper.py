import numpy as np
from bionumpy.kmers import fast_hash

def bionumpy_hash(numeric_sequence, k):
    # reversing stuff to be compatible with old hashing
    kmers = fast_hash(numeric_sequence.astype(np.uint8)[::-1], k)[::-1]
    return kmers





