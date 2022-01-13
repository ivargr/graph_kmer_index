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

    print(index.get_nodes(sequence_to_kmer_hash("AAA")))

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



def test_long_node():
    graph = Graph.from_dicts(
        {1: "ATC", 2: "AAAAAAAA", 3: "T", 4: "CTA"},
        {1: [2, 3], 2: [4], 3: [4]},
        [1, 2, 4]
    )
    graph.set_numeric_node_sequences()
    finder = DenseKmerFinder(graph, k=3, include_reverse_complements=False)
    finder.find()
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)

    assert len(index.get_nodes(sequence_to_kmer_hash("AAA"))) == 6
    assert len(index.get_nodes(sequence_to_kmer_hash("AAC"))) == 2


def test_empty_dummy_nodes():
    graph = Graph.from_dicts(
        {1: "ACT", 2: "C", 3: "", 4: "ACT"},
        {1: [2, 3], 3: [4], 2: [4]},
        [1, 2, 4]
    )
    graph.set_numeric_node_sequences()
    finder = DenseKmerFinder(graph, k=3, include_reverse_complements=False)
    finder.find()
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)

    assert set(index.get_nodes(sequence_to_kmer_hash("CTA"))) == set([1, 3, 4])
    assert set(index.get_nodes(sequence_to_kmer_hash("TCA"))) == set([1, 2, 4])



def test_empty_dummy_nodes2():
    graph = Graph.from_dicts(
        {1: "AAAAA", 2: "", 3: "CCCCCC"},
        {1: [2], 2: [3]},
        [1, 3]
    )
    graph.set_numeric_node_sequences()
    finder = DenseKmerFinder(graph, k=3, include_reverse_complements=False)
    finder.find()
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)

    assert set(index.get_nodes(sequence_to_kmer_hash("ACC"))) == set([1, 2, 3])
    assert set(index.get_nodes(sequence_to_kmer_hash("CCC"))) == set([3])
    assert set(index.get_nodes(sequence_to_kmer_hash("AAC"))) == set([1, 2, 3])
    assert len(index.get_nodes(sequence_to_kmer_hash("AAA"))) == 3
    assert len(index.get_nodes(sequence_to_kmer_hash("CCC"))) == 4


def test_empty_dummy_nodes3():
    graph = Graph.from_dicts(
        {1: "AAAAA", 2: "G", 3: "", 4: "CCCCCC"},
        {1: [2], 2: [3], 3: [4]},
        [1, 2, 4]
    )
    graph.set_numeric_node_sequences()
    finder = DenseKmerFinder(graph, k=3, include_reverse_complements=False)
    finder.find()
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)

    assert set(index.get_nodes(sequence_to_kmer_hash("AGC"))) == set([1, 2, 3, 4])
    assert set(index.get_start_nodes(sequence_to_kmer_hash("AGC"))) == set([4])
    assert set(index.get_start_offsets(sequence_to_kmer_hash("AGC"))) == set([0])
    assert set(index.get_start_offsets(sequence_to_kmer_hash("AAA"))) == set([2, 3, 4])
    assert set(index.get_nodes(sequence_to_kmer_hash("CCC"))) == set([4])


def test_graph_with_multiple_critical_points():
    graph = Graph.from_dicts(
        {1: "CCCCC", 2: "G", 3: "", 4: "ACT", 5: "", 6: "GC", 7: "A", 8: "T", 9: "G", 10: "GGG"},
        {1: [2, 3], 2: [4], 3: [4], 4: [5, 6], 5: [7], 6: [7], 7: [8, 9], 8: [10], 9: [10]},
        [1, 2, 4, 7, 8, 10]
    )
    graph.set_numeric_node_sequences()
    finder = DenseKmerFinder(graph, k=3, include_reverse_complements=False)
    finder.find()
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)

    assert set(index.get_nodes(sequence_to_kmer_hash("TAT"))) == set([4, 5, 7, 8])
    assert set(index.get_start_offsets(sequence_to_kmer_hash("TAT"))) == set([0])
    assert set(index.get_start_nodes(sequence_to_kmer_hash("TAT"))) == set([8])
    assert set(index.get_nodes(sequence_to_kmer_hash("ACT"))) == set([4])
    assert set(index.get_nodes(sequence_to_kmer_hash("GGG"))) == set([9, 10])
    assert set(index.get_nodes(sequence_to_kmer_hash("CAC"))) == set([1, 3, 4])


def test_two_long_nodes():
    graph = Graph.from_dicts(
        {1: "CCCCCCCCCC", 2: "AAAA"},
        {1: [2]},
        [1, 2]
    )
    graph.set_numeric_node_sequences()
    finder = DenseKmerFinder(graph, k=3, include_reverse_complements=False)
    finder.find()
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)

    start_pos = set(index.get_start_offsets(sequence_to_kmer_hash("CCC")))
    assert set(start_pos) == set([2, 3, 4, 5, 6, 7, 8, 9])

    start_pos = set(index.get_start_offsets(sequence_to_kmer_hash("AAA")))
    assert set(start_pos) == set([2, 3])


def test_neighbouring_dummy_nodes():
    pass


very_simple_test()
simple_test()
test_nested_paths()
test_empty_dummy_nodes2()
test_empty_dummy_nodes3()
test_empty_dummy_nodes()
test_graph_with_multiple_critical_points()
test_two_long_nodes()
test_long_node()
