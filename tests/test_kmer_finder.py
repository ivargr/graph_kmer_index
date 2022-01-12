import logging
from graph_kmer_index.kmer_finder import DenseKmerFinder
from graph_kmer_index import KmerIndex2, sequence_to_kmer_hash
from obgraph import Graph
logging.basicConfig(level=logging.INFO)
import numpy as np


def very_simple_test():

    graph = Graph.from_dicts(
        {0: "AAA", 1: "C", 2: "T", 3: "AAA"},
        {0: [1, 2], 2: [3], 1: [3]},
        [0, 1, 3]
    )
    graph.set_numeric_node_sequences()
    finder = DenseKmerFinder(graph, k=3, include_reverse_complements=False)
    finder.find()
    flat = finder.get_flat_kmers()

    index = KmerIndex2.from_flat_kmers(flat)

    assert np.all(index.get_nodes(sequence_to_kmer_hash("ATA")) == [0, 2, 3])
    assert np.all(index.get_start_nodes(sequence_to_kmer_hash("ATA")) == [3, 3, 3])
    assert np.all(index.get_start_offsets(sequence_to_kmer_hash("ATA")) == [0, 0, 0])

    assert set(index.get_nodes(sequence_to_kmer_hash("ACA"))) == set([0, 1, 3])
    assert set(index.get_nodes(sequence_to_kmer_hash("AAA"))) == set([0, 3])

    assert len(index.get_all_kmers()) == 16


def simple_test():

    graph = Graph.from_dicts(
        {0: "ACTGACTG", 1: "A", 2: "T", 3: "AAAAA", 4: "C", 5: "T", 6: "TGGGGG"},
        {0: [1, 2], 2: [3], 1: [3], 3: [4, 5], 4: [6], 5: [6]},
        [0, 1, 3, 4, 6]
    )
    graph.set_numeric_node_sequences()
    finder = DenseKmerFinder(graph, k=3, include_reverse_complements=False)
    finder.find()
    flat = finder.get_flat_kmers()

    index = KmerIndex2.from_flat_kmers(flat)

    assert set(index.get_nodes(sequence_to_kmer_hash("ACT"))) == set([0, 0, 3, 4, 6])
    assert set(index.get_start_nodes(sequence_to_kmer_hash("AAC"))) == set([4])
    assert set(index.get_start_offsets(sequence_to_kmer_hash("AAC"))) == set([0])



def test_nested_paths():
    # checking that the recusion stops at the nested node 3 so that we are not duplicating entires after that
    graph = Graph.from_dicts(
        {0: "AAA", 1: "C", 2: "T", 3: "AAAA", 4: "C", 5: "G", 6: "AAA", 7: "TTT"},
        {0: [1, 2, 7], 1: [3], 2: [3], 3: [4, 5], 4: [6], 5: [6], 7: [6]},
        [0, 1, 3, 4, 6]
    )
    graph.set_numeric_node_sequences()
    finder = DenseKmerFinder(graph, k=3, include_reverse_complements=False)
    finder.find()
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)
    assert len(flat._hashes) == 41, len(flat._hashes)

very_simple_test()
simple_test()
test_nested_paths()