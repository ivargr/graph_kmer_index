import logging
from graph_kmer_index.kmer_finder import DenseKmerFinder
from graph_kmer_index import KmerIndex2, sequence_to_kmer_hash
from obgraph import Graph
logging.basicConfig(level=logging.INFO)
import numpy as np
from graph_kmer_index import kmer_hash_to_sequence


def very_simple_test():

    graph = Graph.from_dicts(
        {0: "AAA", 1: "C", 2: "T", 3: "AAA"},
        {0: [1, 2], 2: [3], 1: [3]},
        [0, 1, 3]
    )
    finder = DenseKmerFinder(graph, k=3)
    finder.find()
    flat = finder.get_flat_kmers()
    print("Kmers found: %s" % finder.kmers_found)

    index = KmerIndex2.from_flat_kmers(flat, modulo=15)

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
    finder = DenseKmerFinder(graph, k=3)
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
    finder = DenseKmerFinder(graph, k=3)
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
    finder = DenseKmerFinder(graph, k=3)
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
    finder = DenseKmerFinder(graph, k=3)
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
    finder = DenseKmerFinder(graph, k=3)
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
    finder = DenseKmerFinder(graph, k=3)
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
    finder = DenseKmerFinder(graph, k=3)
    finder.find()
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)

    assert set(index.get_nodes(sequence_to_kmer_hash("TAT"))) == set([4, 5, 7, 8])
    assert set(index.get_start_offsets(sequence_to_kmer_hash("TAT"))) == set([0])
    assert set(index.get_start_nodes(sequence_to_kmer_hash("TAT"))) == set([8])
    assert set(index.get_nodes(sequence_to_kmer_hash("ACT"))) == set([4])
    assert set(index.get_nodes(sequence_to_kmer_hash("GGG"))) == set([9, 10])
    assert set(index.get_nodes(sequence_to_kmer_hash("CAC"))) == set([1, 3, 4])


def test_two_long_nodes1():
    graph = Graph.from_dicts(
        {1: "CCCCCCCCCC", 2: "AAAA"},
        {1: [2]},
        [1, 2]
    )
    finder = DenseKmerFinder(graph, k=3)
    finder.find()
    flat = finder.get_flat_kmers()
    print(flat)
    index = KmerIndex2.from_flat_kmers(flat)

    start_pos = set(index.get_start_offsets(sequence_to_kmer_hash("CCC")))
    assert set(start_pos) == set([2, 3, 4, 5, 6, 7, 8, 9])

    start_pos = set(index.get_start_offsets(sequence_to_kmer_hash("AAA")))
    assert set(start_pos) == set([2, 3])


def test_two_long_nodes2():
    graph = Graph.from_dicts(
        {1: "CATGCATGCCTG", 2: "CCAAG"},
        {1: [2]},
        [1, 2]
    )
    finder = DenseKmerFinder(graph, k=5)
    finder.find()
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)

    start_pos = set(index.get_start_offsets(sequence_to_kmer_hash("CTGCC")))
    assert set(start_pos) == set([1])
    print(index.get_start_nodes(sequence_to_kmer_hash("CTGCC")))
    assert set(index.get_start_nodes(sequence_to_kmer_hash("CTGCC"))) == set([2])
    assert set(index.get_nodes(sequence_to_kmer_hash("CTGCC"))) == set([1, 2])
    assert len(index.get_start_nodes(sequence_to_kmer_hash("CTGCC"))) == 2

    assert list(index.get_start_offsets(sequence_to_kmer_hash("GCCTG"))) == [11]
    assert list(index.get_start_offsets(sequence_to_kmer_hash("CCAAG"))) == [4]
    assert set(list(index.get_start_offsets(sequence_to_kmer_hash("CATGC")))) == set([4, 8])




def test_neighbouring_dummy_nodes():

    graph = Graph.from_dicts(
        {1: "ACT", 2: "", 3: "GGG", 4: "", 5: "A", 6: "CCC"},
        {1: [2, 3], 2: [4, 5], 3: [4, 5], 4: [6], 5: [6]},
        [1, 5, 6]
    )

    finder = DenseKmerFinder(graph, k=3)
    finder.find()
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)


    # passing two dummy nodes:
    assert set(index.get_nodes(sequence_to_kmer_hash("TCC"))) == set([1, 2, 4, 6])
    # single dummy node
    assert set(index.get_nodes(sequence_to_kmer_hash("TAC"))) == set([1, 2, 5, 6])
    assert set(index.get_nodes(sequence_to_kmer_hash("GCC"))) == set([3, 4, 6])


def test_max_variant_nodes():
    graph = Graph.from_dicts(
        {1: "ACT", 2: "", 3: "GGG", 4: "", 5: "A", 6: "CCC"},
        {1: [2, 3], 2: [4, 5], 3: [4, 5], 4: [6], 5: [6]},
        [1, 5, 6]
    )

    max_variant_nodes = 0
    finder = DenseKmerFinder(graph, k=3, max_variant_nodes=max_variant_nodes)
    finder.find()
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)

    assert set(index.get_nodes(sequence_to_kmer_hash("CTA"))) == set([1, 2, 5])
    assert set(index.get_nodes(sequence_to_kmer_hash("TAC"))) == set([1, 2, 5, 6])
    assert set(index.get_nodes(sequence_to_kmer_hash("GGG"))) == set([])
    assert set(index.get_nodes(sequence_to_kmer_hash("TCC"))) == set([])


    max_variant_nodes = 1  # first variant node should be chosen, but not first AND second
    finder = DenseKmerFinder(graph, k=3, max_variant_nodes=max_variant_nodes)
    finder.find()
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)

    assert set(index.get_nodes(sequence_to_kmer_hash("TGG"))) == set([1, 3])
    assert set(index.get_nodes(sequence_to_kmer_hash("TCC"))) == set([1, 2, 4, 6])
    assert set(index.get_nodes(sequence_to_kmer_hash("GCC"))) == set()
    assert set(index.get_nodes(sequence_to_kmer_hash("GGC"))) == set()
    assert set(index.get_nodes(sequence_to_kmer_hash("GAC"))) == set([3, 5, 6])



def test_snp_and_long_node():
    graph = Graph.from_dicts(
        {1: "ACTACTACTACT", 2: "G", 3: "C", 4: "GCAGCA"},
        {1: [2, 3], 2: [4], 3: [4]},
        [1, 2, 4]
    )

    finder = DenseKmerFinder(graph, k=3)
    finder.find()
    flat = finder.get_flat_kmers()
    print(flat)
    index = KmerIndex2.from_flat_kmers(flat)

    assert set(index.get_start_offsets(sequence_to_kmer_hash("CTG"))) == set([0])
    assert set(index.get_start_offsets(sequence_to_kmer_hash("TAC"))) == set([4, 7, 10])



def test_large_k():
    graph = Graph.from_dicts(
        {1: "G"*100, 2: "C", 3: "T", 4: "G"*10},
        {1: [2, 3], 2: [4], 3: [4]},
        [1, 2, 4]
    )

    finder = DenseKmerFinder(graph, k=31)
    finder.find()
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)
    print(index.get_start_offsets(sequence_to_kmer_hash("G"*31)))


def test_find_kmers_from_position():
    graph = Graph.from_dicts(
        {1: "ACTACT", 2: "G", 3: "C", 4: "GCAGCA"},
        {1: [2, 3], 2: [4], 3: [4]},
        [1, 2, 4]
    )

    finder = DenseKmerFinder(graph, k=3, only_store_nodes=set([2, 3]))
    finder.find_only_kmers_starting_at_position(1, 4)
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)

    assert np.all(index.get_nodes(sequence_to_kmer_hash("CTG")) == [2])
    assert np.all(index.get_nodes(sequence_to_kmer_hash("CTC")) == [3])

    finder = DenseKmerFinder(graph, k=5, only_store_nodes=set([2, 3]))
    finder.find_only_kmers_starting_at_position(1, 5)
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)
    print(finder.kmers_found)

    assert np.all(index.get_nodes(sequence_to_kmer_hash("TGGCA")) == [2])
    assert np.all(index.get_nodes(sequence_to_kmer_hash("TCGCA")) == [3])
    #assert set(index.get_start_offsets(sequence_to_kmer_hash("CTG"))) == set([0])



def test_special_case():
    graph = Graph.from_dicts(
        {1: "taacccctaacccctaaccctaaccctaac",
         2: "", 3: "G", 4: "ccctaaccctaaccctaacccctaacccta"},
        {1: [2, 3], 2: [4], 3: [4]},
        [1, 4]
    )

    finder = DenseKmerFinder(graph, k=31, only_store_nodes=set([2, 3]))
    finder.find_only_kmers_starting_at_position(1, 22)
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)

    print(flat._hashes)

    search = "accctaacccctaaccctaaccctaacccct"
    hash = sequence_to_kmer_hash(search)
    assert np.all(index.get_start_offsets(hash) == [22])
    assert np.all(index.get_start_nodes(hash) == [4])



def test_indel():
    graph = Graph.from_dicts(
        {1: "ACTGA",
         2: "", 3: "C", 4: "GGGGGGGGG"},
        {1: [2, 3], 2: [4], 3: [4]},
        [1, 4]
    )

    finder = DenseKmerFinder(graph, k=9, only_store_nodes=set([2, 3]))
    finder.find_only_kmers_starting_at_position(1, 2)
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)

    assert np.all(index.get_nodes(sequence_to_kmer_hash("TGAGGGGGG")) == [2])
    assert np.all(index.get_nodes(sequence_to_kmer_hash("TGACGGGGG")) == [3])

def test_snp_and_indel():
    graph = Graph.from_dicts(
        {1: "ACTGAACTG",
         2: "A", 3: "C", 4: "GGGG",
         5: "", 6: "T", 7: "CCCCCC"},
        {1: [3, 2], 2: [4], 3: [4], 4: [5, 6], 5: [7], 6: [7]},
        [1, 2, 4, 6, 7]
    )

    finder = DenseKmerFinder(graph, k=13, only_store_nodes=set([5, 6]), max_variant_nodes=5)
    finder.find_only_kmers_starting_at_position(1, 6)
    flat = finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)

    assert np.all(index.get_nodes(sequence_to_kmer_hash("CTGAGGGGCCCCC")) == [5])
    assert np.all(index.get_nodes(sequence_to_kmer_hash("CTGAGGGGTCCCC")) == [6])



def test_some_case():
    graph = Graph.from_dicts(
        {1: "AAAAAACTG", 2: "A", 3: "G", 4: "GC", 5: "T", 6: "C", 7: "TGAGCCCCC",
         8: "A", 9: "T", 10: "AAAAA"},
        {1: [2, 3], 2: [4], 3: [4], 4: [5, 6], 5: [7], 6: [7], 7: [8, 9], 9: [10], 8: [10]},
        [1, 2, 4, 5, 7, 8, 10]
    )

    kmer_finder = DenseKmerFinder(graph, k=5)
    kmer_finder.find()
    flat = kmer_finder.get_flat_kmers()
    index = KmerIndex2.from_flat_kmers(flat)

    print(index.get_start_nodes(sequence_to_kmer_hash("CTGAG")))
    assert set(index.get_start_nodes(sequence_to_kmer_hash("CTGAG"))) == set([4, 7])


def test_case2():
    graph = Graph.from_dicts(
        {0: "AGTAGA", 1: "G", 2: "CT", 3: "A", 4: "CTA", 5: "G", 6: "A", 7: "TCATA"},
        {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: [5, 6], 5: [7], 6: [7], 7: []},
        [0, 1, 3, 4, 5, 7]
    )

    kmer_finder = DenseKmerFinder(graph, k=3)
    kmer_finder.find()
    kmers, nodes = kmer_finder.get_found_kmers_and_nodes()


def test_case1():
    """
    graph = Graph.from_dicts(
        {0: "AGTAGA", 1: "G", 2: "CT", 3: "A", 4: "CTA", 5: "G", 6: "A", 7: "TCATA"},
        {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: [5, 6], 5: [7], 6: [7], 7: []},
        [0, 1, 3, 4, 5, 7]
    )
    """

    graph = Graph.from_dicts(
        {0: "AGTAGA", 1: "G", 2: "CT", 3: "ACTA", 5: "G", 6: "A", 7: "TCATA"},
        {0: [1, 2], 1: [3], 2: [3], 3: [5, 6], 5: [7], 6: [7], 7: []},
        [0, 1, 3, 5, 7]
    )

    kmer_finder = DenseKmerFinder(graph, k=3)
    kmer_finder.find()
    kmers, nodes = kmer_finder.get_found_kmers_and_nodes()

    correct = [
        ["AGT", 0],
        ["GTA", 0],
        ["TAG", 0],
        ["AGA", 0],
        ["GAG", 0],
        ["GAG", 1],
        ["AGA", 0],
        ["AGA", 1],
        ["AGA", 3],
        ["GAC", 1],
        ["GAC", 3],
        ["GAC", 0],
        ["GAC", 2],
        ["ACT", 0],
        ["ACT", 2],
        ["CTA", 2],
        ["CTA", 3],
        ["TAC", 2],
        ["TAC", 3],
        ["ACT", 3],
        ["CTA", 3],
        ["TAG", 3],
        ["TAG", 5],
        ["AGT", 3],
        ["AGT", 5],
        ["AGT", 7],
        ["GTC", 5],
        ["GTC", 7],
        ["TAA", 3],
        ["TAA", 6],
        ["AAT", 3],
        ["AAT", 6],
        ["AAT", 7],
        ["ATC", 6],
        ["ATC", 7],
        ["TCA", 7],
        ["CAT", 7],
        ["ATA", 7]
    ]

    for i, (kmer, node) in enumerate(zip(kmers, nodes)):
        print(kmer_hash_to_sequence(kmer, 3).upper(), node, *correct[i])
        assert kmer_hash_to_sequence(kmer, 3).upper() == correct[i][0]
        assert node == correct[i][1]

    print(len(correct))

"""
very_simple_test()
simple_test()
test_nested_paths()
test_empty_dummy_nodes2()
test_empty_dummy_nodes3()
test_empty_dummy_nodes()
test_graph_with_multiple_critical_points()
test_neighbouring_dummy_nodes()
test_max_variant_nodes()
test_long_node()
test_two_long_nodes()
test_two_long_nodes2()
test_snp_and_long_node()
test_large_k()
test_find_kmers_from_position()
test_special_case()
test_indel()
test_snp_and_indel()
test_some_case()
"""
#test_case2()