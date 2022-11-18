import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from graph_kmer_index.structural_variants import sample_kmers_from_structural_variants
from graph_kmer_index import letter_sequence_to_numeric
from obgraph.variant_to_nodes import VariantToNodes
from graph_kmer_index import KmerIndex
from graph_kmer_index import sequence_to_kmer_hash


class DummyGraph:
    def __init__(self, node_sequences: dict):
        self.node_sequences = node_sequences

    def get_numeric_node_sequence(self, node):
        return letter_sequence_to_numeric(self.node_sequences[node])

    def get_node_size(self, node):
        return len(self.node_sequences[node])


class DummyKmerIndex:
    def get_frequency(self, kmer):
        return 1


def test_sample_kmers_from_structural_variants():
    graph = DummyGraph({
        1: "AAAAAAAAAAA",
        2: "ACTG",
        3: "GGGGAAAACCCCAAAA",
        4: "AGGGG"
    })

    variant_to_nodes = VariantToNodes(
        np.array([1, 3]), np.array([2, 4])
    )

    kmers = sample_kmers_from_structural_variants(graph, variant_to_nodes,
                                                  DummyKmerIndex(), k=5)

    logging.info(kmers.describtion())

    index = KmerIndex.from_flat_kmers(kmers)

    assert np.all(index.get_nodes(sequence_to_kmer_hash("AAAAA")) == [1])
    print("NODES", index.get_nodes(sequence_to_kmer_hash("GGGGA")))
    assert np.all(index.get_nodes(sequence_to_kmer_hash("GGGGA")) == [3])
    assert np.all(index.get_nodes(sequence_to_kmer_hash("AAACC")) == [3])

