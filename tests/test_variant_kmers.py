import logging
logging.basicConfig(level=logging.INFO)
import pytest
import numpy as np
from graph_kmer_index.unique_variant_kmers import UniqueVariantKmersFinder
from obgraph import Graph
from obgraph.variant_to_nodes import VariantToNodes
from obgraph.variants import VcfVariants, VcfVariant
from graph_kmer_index.kmer_finder import DenseKmerFinder
from graph_kmer_index import KmerIndex
from graph_kmer_index import sequence_to_kmer_hash
from graph_kmer_index.reverse_kmer_index import ReverseKmerIndex


@pytest.fixture
def k():
    return 5

@pytest.fixture
def graph():
    graph = Graph.from_dicts(
        {1: "AAAAAACTG", 2: "A", 3: "G", 4: "GC", 5: "T", 6: "C", 7: "TGAGCCCCC",
         8: "", 9: "TC", 10: "AAAAA"},
        {1: [2, 3], 2: [4], 3: [4], 4: [5, 6], 5: [7], 6: [7], 7: [8, 9], 9: [10], 8: [10]},
        [1, 2, 4, 5, 7, 10]
    )
    return graph


@pytest.fixture
def variant_to_nodes():
    return VariantToNodes(np.array([2, 5, 8]), np.array([3, 6, 9]))


@pytest.fixture
def variants():
    return VcfVariants([
        VcfVariant(1, 10, "A", "G", vcf_line_number=0, type="SNP"),
        VcfVariant(1, 13, "T", "C", vcf_line_number=1, type="SNP"),
        VcfVariant(1, 22, "C", "CTC", vcf_line_number=2, type="INDEL"),
    ])


@pytest.fixture
def kmer_index_with_frequencies(graph, k):
    kmer_finder = DenseKmerFinder(graph, k)
    kmer_finder.find()
    flat_kmers = kmer_finder.get_flat_kmers(v="1")
    kmer_index_with_frequencies = KmerIndex.from_flat_kmers(flat_kmers)
    assert kmer_index_with_frequencies.get_frequency(sequence_to_kmer_hash("CTGAG")) == 2
    return kmer_index_with_frequencies

@pytest.fixture
def position_id_index(graph):
    from obgraph.position_id import PositionId
    return PositionId.from_graph(graph)


@pytest.fixture
def kmer_finder(graph, variant_to_nodes, variants, k, position_id_index, kmer_index_with_frequencies):
    finder = UniqueVariantKmersFinder(graph, variant_to_nodes, variants, k=5,
                                      kmer_index_with_frequencies=kmer_index_with_frequencies,
                                      use_dense_kmer_finder=True, position_id_index=position_id_index
                                      )
    return finder


def test_kmers_from_position(kmer_finder):
    kmers_found = kmer_finder.find_unique_kmers()
    index = KmerIndex.from_flat_kmers(kmers_found)
    nodes = index.get_nodes(sequence_to_kmer_hash("CTGAG"))
    assert nodes is None, "CTGAG should not be in index"
    reverse_index = ReverseKmerIndex.from_flat_kmers(kmers_found)
    assert len(reverse_index.get_node_kmers(2)) > 0
    assert len(reverse_index.get_node_kmers(3)) > 0


def test_find_variant_kmers_for_node(kmer_finder, variants):

    variant = variants[0]
    results = kmer_finder.find_kmers_over_variant_node(variant, 3)
    index = KmerIndex.from_flat_kmers(results)
    assert np.all(index.get_nodes(sequence_to_kmer_hash("GGCTT")) == [3])
    assert np.all(index.get_nodes(sequence_to_kmer_hash("GGCCT")) == [3])

    results = kmer_finder.find_kmers_over_variant_node(variant, 2)
    index = KmerIndex.from_flat_kmers(results)
    assert 2 in index.get_nodes(sequence_to_kmer_hash("AGCCT"))
    assert 2 in index.get_nodes(sequence_to_kmer_hash("AGCCT"))

    # indel
    results = kmer_finder.find_kmers_over_variant(variants[2], 8, 9)
    index = KmerIndex.from_flat_kmers(results)
    assert 8 in index.get_nodes(sequence_to_kmer_hash("CAAAA"))
    assert 9 in index.get_nodes(sequence_to_kmer_hash("TCAAA"))

