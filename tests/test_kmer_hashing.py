from graph_kmer_index import sequence_to_kmer_hash, letter_sequence_to_numeric, kmer_hash_to_sequence
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
    assert hash1 != hash2
    assert hash1 == hash2 * 4 + 2


    # Testing for overflow
    print(sequence_to_kmer_hash("G"*31))
    assert sequence_to_kmer_hash("G"*31) == 4611686018427387903


def test_hash_and_reverse():
    sequence = "atg"
    hash = sequence_to_kmer_hash(sequence)
    print(hash)
    sequence2 = kmer_hash_to_sequence(hash, len(sequence))
    print(sequence2)

    sequence = "ggtagctctcgccagctcctagaaggagga"
    hash = sequence_to_kmer_hash(sequence)
    print(hash)
    sequence2 = kmer_hash_to_sequence(hash, len(sequence))
    print(sequence, sequence2, hash, sequence_to_kmer_hash(sequence2))


    # todo:
    assert sequence2 == sequence, "%s != %s" % (sequence2, sequence)
