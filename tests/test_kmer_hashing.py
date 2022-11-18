from graph_kmer_index import sequence_to_kmer_hash, letter_sequence_to_numeric, kmer_hash_to_sequence
import numpy as np
from graph_kmer_index.kmer_hashing import kmer_hash_to_reverse_complement_hash, kmer_hashes_to_reverse_complement_hash
from graph_kmer_index import ReadKmers
from Bio.Seq import Seq


def test():
    assert sequence_to_kmer_hash("ACTG") == 0 + 1 * 16 + 3 * 4 + 2
    seq1 =  "CAtgAACAtttggtAATCTACAtgAACAttt"
    seq3 =  "CAtgAACAtttggtAATCTACAtgAACAtta"
    seq2 = "ACAtgAACAtttggtAATCTACAtgAACAtt"
    hash1 = sequence_to_kmer_hash(seq1)
    hash2 = sequence_to_kmer_hash(seq2)
    hash3 = sequence_to_kmer_hash(seq3)
    assert sequence_to_kmer_hash(seq1) == np.sum(np.power(4, np.arange(0, 31)[::-1]) * letter_sequence_to_numeric(seq1))
    assert hash1 != hash2
    assert hash1 == hash2 * 4 + 3


    # Testing for overflow
    print(sequence_to_kmer_hash("T"*31))
    assert sequence_to_kmer_hash("T"*31) == 4611686018427387903


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


def test_rev_comp_hash():
    sequences = ["AcATaCAG",
                 "AGACATTA",
                 "GGGGAAAACCCCTTTTAAAACCCCTTTTGGG",
                 "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",
                 "ACT"]

    for seq in sequences:
        k = len(seq)
        hash = sequence_to_kmer_hash(seq)
        rev_comp_hash = kmer_hash_to_reverse_complement_hash(hash, k)
        back_to_original_hash = kmer_hash_to_reverse_complement_hash(rev_comp_hash, k)
        assert hash == back_to_original_hash
        assert kmer_hash_to_sequence(rev_comp_hash, k).lower() == str(Seq(seq).reverse_complement()).lower()


def test_rev_comp_hashes():
    sequences = ["ACACTTACG",
                 "acgactaca",
                 "AATTGGGGG",
                 "ACACACACT"]
    k = len(sequences[0])
    hashes = np.array([sequence_to_kmer_hash(sequence) for sequence in sequences])
    reverse_complement = kmer_hashes_to_reverse_complement_hash(hashes, k)
    back = kmer_hashes_to_reverse_complement_hash(reverse_complement, k)
    assert np.all(back == hashes)




test_rev_comp_hash()
test_rev_comp_hashes()