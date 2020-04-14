from alignment_free_graph_genotyper.chain_genotyper import read_kmers
import pyximport; pyximport.install(language_level=3)
from graph_kmer_index.cython_kmer_index import get_nodes_and_ref_offsets_from_multiple_kmers
from graph_kmer_index import KmerIndex
import numpy as np
from graph_kmer_index.flat_kmers import letter_sequence_to_numeric
import time

power_array = np.power(4, np.arange(0, 31))
read = "GTCTTCCGAGCGTCAGGCCGCCCCTACCCGTGCTTTCTGCTCTGCAGACCCTCTTCCTAGACCTCCGTCCTTTGTCCCATCGCTGCCTTCCCCTCAAGCTCAGGGCCAAGCTGTCCGCCAACCTCGGCTCCTCCGGGCAGCCCTCGCCCG"
read = "CACCTGGGATCTGAGGCTGCCTCAAAAGGCAGCACAGGCGATGCCGGGTGCACAGGGTGGCGGGTGCCCCGGACTTCATGGTAATGGTGGGGCTGGGGAAGGGCCTGAAGCTCTGGCCCCTGTGGGAGCTCCTGCTGTGTTCTGGGGGCA"
kmers = read_kmers(read, power_array)


index = KmerIndex.from_file("merged31")
print(type(index._nodes))
print(type(index._ref_offsets))
print("Done reading index")

def test_cython():
    nodes, ref_offsets, read_offsets = get_nodes_and_ref_offsets_from_multiple_kmers(
            kmers,
            index._hasher._hashes,
            index._hashes_to_index,
            index._n_kmers,
            index._nodes,
            index._ref_offsets
        )


def test_noncython():
    nodes, ref_offsets, read_offsets = index.get_nodes_and_ref_offsets_from_multiple_kmers(kmers)

test_cython()
test_noncython()


t0 = time.time()
for i in range(0, 10000):
    test_cython()

t1 = time.time()
print(t1-t0)

