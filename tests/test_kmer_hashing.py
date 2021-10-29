from graph_kmer_index import sequence_to_kmer_hash, letter_sequence_to_numeric
import numpy as np


def test():
    assert sequence_to_kmer_hash("ACTG") == 0 + 1 * 16 + 2 * 4 + 3
    seq1 =  "CAtgAACAtttggtAATCTACAtgAACAttt"
    seq3 =  "CAtgAACAtttggtAATCTACAtgAACAtta"
    seq2 = "ACAtgAACAtttggtAATCTACAtgAACAtt"
    hash1 = sequence_to_kmer_hash(seq1)
    hash2 = sequence_to_kmer_hash(seq2)
    hash3 = sequence_to_kmer_hash(seq3)
    assert sequence_to_kmer_hash(seq1) == np.sum(np.power(4, np.arange(0, 31)[::-1]) * letter_sequence_to_numeric(seq1))

    print(letter_sequence_to_numeric(seq1))
    print(letter_sequence_to_numeric(seq2))
    print(letter_sequence_to_numeric(seq3))
    print(len(seq1))
    print("hash 3:", hash3)
    print("hash 2:", hash2)
    print("hash 1:", hash1)
    print(hash2 * 4)
    assert hash1 != hash2
    assert hash1 == hash2 * 4 + 2


    # Testing for overflow
    print(sequence_to_kmer_hash("G"*31))
    assert sequence_to_kmer_hash("G"*31) == 4611686018427387903

