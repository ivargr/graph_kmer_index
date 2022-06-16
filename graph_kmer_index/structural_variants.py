import logging
from .bionumpy_wrapper import bionumpy_hash
from .flat_kmers import FlatKmers
import numpy as np

def sample_kmers_from_structural_variants(graph, variant_to_nodes, kmer_index_with_frequencies, k, max_frequency=2):
    """"
    For every variant with big nodes,
    try to find kmers within the node with low frequency in kmer_index_with_frequencies
    """

    kmers_found = []
    nodes_found = []
    ref_offsets_found = []

    for ref_node, var_node  in variant_to_nodes:
        for node in (ref_node, var_node):
            if graph.get_node_size(node) > k + 5:
                node_sequence = graph.get_numeric_node_sequence(node)
                node_kmers = bionumpy_hash(node_sequence, k)
                kmer_frequencies = np.array([kmer_index_with_frequencies.get_frequency(k) for k in node_kmers])
                valid = np.where(kmer_frequencies < max_frequency)[0]

                # do not choose overlapping kmers
                chosen = []
                prev = -10000
                for v in valid:
                    if v >= prev + k:
                        chosen.append(v)
                        prev = v

                #logging.info("Node kmers: %s" % node_kmers)
                #logging.info("Chose positions %s with kmers %s for node %d" % (chosen, node_kmers[chosen], node))

                if len(chosen) > 0:
                    kmers_found.extend(node_kmers[chosen])
                    nodes_found.extend([node]*len(chosen))
                    ref_offsets_found.extend([0]*len(chosen))


    return FlatKmers(np.array(kmers_found, dtype=np.uint64),
                     np.array(nodes_found, dtype=np.uint32),
                     np.array(ref_offsets_found, dtype=np.uint32))