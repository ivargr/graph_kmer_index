import logging
logging.basicConfig(level=logging.INFO)
from graph_kmer_index.snp_kmer_finder import SnpKmerFinder
from graph_kmer_index import KmerIndex
from obgraph import Graph
from graph_kmer_index import sequence_to_kmer_hash


def test_simple_snp_graph():

    graph = Graph.from_dicts(
        {1: "ACTG", 2: "A", 3: "G", 4: "AAAT"},
        {1: [2, 3],
         2: [4],
         3: [4]},
        [1, 2, 4]
    )

    kmer_finder = SnpKmerFinder(graph, k=3)
    flat_kmers = kmer_finder.find_kmers()
    print(kmer_finder.kmers_found)
    print(flat_kmers._ref_offsets)
    print(flat_kmers._nodes)
    print(flat_kmers._hashes)

    assert kmer_finder.has_kmer("ACT", {1})
    assert kmer_finder.has_kmer("GAA", {1, 2, 4})
    assert kmer_finder.has_kmer("GGA", {1, 3, 4})
    assert kmer_finder.has_kmer("AAT", {4})


def test_indel_graph():
    graph = Graph.from_dicts(
        {1: "ACTG", 2: "A", 3: "", 4: "TAAT"},
        {1: [2, 3],
         2: [4],
         3: [4]},
        [1, 2, 4]
    )
    kmer_finder = SnpKmerFinder(graph, k=3)
    flat_kmers = kmer_finder.find_kmers()
    print(kmer_finder.kmers_found)

    index = KmerIndex.from_flat_kmers(flat_kmers)
    hits = index.get(sequence_to_kmer_hash("GTA"))
    assert list(hits[1] == [1, 3, 4])
    print(hits)
    hits = index.get(sequence_to_kmer_hash("GAT"))
    assert list(hits[1] == [1, 2, 4])
    print(hits)

def test_indel_graph2():
    graph = Graph.from_dicts(
        {1: "gggggaggcttgtggttagcagagagtgggtggaagacagaggtttgag",
         2: "ga",
         3: "gagagagacccaggggagaaaaccagctgcagaggcaggaggggtccagggcagcccgaggccagagatgggcgtcttccttacagccacctgtggtccc",
         100: ""},
        {
            1: [2, 100],
            2: [3],
            100: [3]
        },
        [1, 2, 3]
    )
    kmer_finder = SnpKmerFinder(graph, k=31)
    flat_kmers = kmer_finder.find_kmers()
    print(kmer_finder.kmers_found)

test_indel_graph2()
#test_simple_snp_graph()
