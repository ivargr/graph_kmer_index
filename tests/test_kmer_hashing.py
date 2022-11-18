from graph_kmer_index import sequence_to_kmer_hash, letter_sequence_to_numeric, kmer_hash_to_sequence
import numpy as np
from graph_kmer_index.kmer_hashing import kmer_hash_to_reverse_complement_hash, kmer_hashes_to_reverse_complement_hash
from graph_kmer_index import ReadKmers
from Bio.Seq import Seq
from graph_kmer_index.kmer_hashing import power_array, reverse_power_array, kmer_hashes_to_bases
from graph_kmer_index.flat_kmers import numeric_to_letter_sequence


def test_simple():
    assert sequence_to_kmer_hash("ACTG") == 0*1 + 1*4 + 3*16 + 2*64  #  + 1 * 16 + 3 * 4 + 2

def test_overflow_issues():
    # sequences that can cause problems if dtypes are not uint64
    seq1 =   "CAtgAACAtttggtAATCTACAtgAACAttt"
    seq2 =  "ACAtgAACAtttggtAATCTACAtgAACAtt"
    seq3 =  "CAtgAACAtttggtAATCTACAtgAACAtta"
    hash1 = sequence_to_kmer_hash(seq1)
    hash2 = sequence_to_kmer_hash(seq2)
    hash3 = sequence_to_kmer_hash(seq3)

    for s in [seq1, seq2, seq3]:
        assert sequence_to_kmer_hash(s) == np.sum(reverse_power_array(31) * letter_sequence_to_numeric(s))

    # Testing for overflow
    print(sequence_to_kmer_hash("T"*31))
    assert sequence_to_kmer_hash("T"*31) == 4611686018427387903


def test_hash_and_reverse():
    sequences = ["atg", "Acacatacgactacg", "CAtgAACAtttggtAATCTACAtgAACAttt", "G"]
    for sequence in sequences:
        hash = sequence_to_kmer_hash(sequence)
        sequence2 = kmer_hash_to_sequence(hash, len(sequence))
        assert sequence2.lower() == sequence.lower(), "%s != %s" % (sequence2, sequence)


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
        rev_comp_sequence = kmer_hash_to_sequence(rev_comp_hash, k).lower()
        print("Sequence: %s" % seq)
        print("Rev comp: %s" % rev_comp_sequence)
        assert rev_comp_sequence == str(Seq(seq).reverse_complement()).lower()


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


def test_kmer_hashes_to_bases():
    sequences = ["ACTG", "TGGC"]
    hashes = np.array([sequence_to_kmer_hash(s) for s in sequences])
    bases = kmer_hashes_to_bases(hashes, 4)

    back = [''.join(numeric_to_letter_sequence(b)).upper() for b in bases]
    assert back == sequences
