import logging
logging.basicConfig(level=logging.INFO)
import sys
from obgraph import Graph
from graph_kmer_index.kmer_finder import DenseKmerFinder
from graph_kmer_index.collision_free_kmer_index import KmerIndex2
from shared_memory_wrapper import to_file, from_file

critical_graph_paths = from_file("critical_paths")

graph = Graph.from_file("obgraph.npz")

kmer_finder = DenseKmerFinder(graph, 31, critical_graph_paths=critical_graph_paths, max_variant_nodes=5,
                              include_reverse_complements=False,
                              only_save_one_node_per_kmer=True,
                              stop_at_critical_path_number=None)
kmer_finder.find()

logging.info("Making index from flat kmers")