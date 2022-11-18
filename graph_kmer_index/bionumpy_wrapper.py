import numpy as np
import bionumpy as bnp

def bionumpy_hash(numeric_sequence, k):
    # reversing stuff to be compatible with old hashing
    #kmers = fast_hash(numeric_sequence.astype(np.uint8)[::-1], k)[::-1]
    numeric_sequence = numeric_sequence   # [::-1]
    encoded_sequence = bnp.EncodedArray(numeric_sequence, bnp.encodings.alphabet_encoding.ACTGEncoding)
    kmers = bnp.sequence.get_kmers(encoded_sequence, k).raw()   # [::-1]  # raw to get numpy array
    return kmers





