import logging
import numpy as np
from .flat_kmers import FlatKmers2
from .critical_graph_paths import CriticalGraphPaths
from npstructures.numpylist import NumpyList


class DenseKmerFinder:
    """
    Finds all possible kmers in graph
    """
    def __init__(self, graph, k, critical_graph_paths=None, only_save_one_node_per_kmer=False, max_variant_nodes=5,
                 include_reverse_complements=False, variant_to_nodes=None, only_store_variant_nodes=False):
        self._graph = graph
        self._linear_ref_nodes = self._graph.linear_ref_nodes()
        self._k = k
        self._only_save_one_node_per_kmer = only_save_one_node_per_kmer
        self._max_variant_nodes = max_variant_nodes
        self._start_nodes = []
        self._start_offsets = []
        self._nodes = []
        self._kmers = []

        self._current_search_start_node = self._graph.get_first_node()
        self._current_search_start_offset = 0
        self._current_hash = 0
        self._current_reverse_hash = 0
        self._current_bases = NumpyList()
        self._current_nodes = NumpyList()
        self._allele_frequencies = NumpyList()

        self.results = []

        self._n_nodes_skipped_because_too_complex = 0
        self._include_reverse_complement = include_reverse_complements

        self._only_store_variant_nodes = only_store_variant_nodes
        self._variant_to_nodes = variant_to_nodes
        if self._only_store_variant_nodes:
            assert variant_to_nodes is not None

        self._critical_graph_paths = critical_graph_paths
        if self._critical_graph_paths is None:
            logging.info("Making critical graph paths since it's not specified. "
                         "Will be faster if critical graph paths is premade")
            self._critical_graph_paths = CriticalGraphPaths.from_graph(graph, k)


        self._positions_treated = set()

    def get_flat_kmers(self):
        return FlatKmers2(np.array(self._kmers, dtype=np.int64), np.array(self._start_nodes, np.int32),
                          np.array(self._start_offsets, dtype=np.int16), np.array(self._nodes, np.int32),
                          np.array(self._allele_frequencies, np.single))

    def _find_search_starting_points(self):
        # finds all critical nodes with length >= k in order to find suitable starting points for recursive searches
        pass

    def _add_kmer(self, kmer, reverse_kmer, start_node, start_offset):
        nodes = self._current_nodes[-self._k:]
        logging.info("Adding kmer %d at node/offset %d/%d with nodes %s. Start node/offset: %d/%d" %
                     (kmer, start_node, start_offset, nodes, start_node, start_offset))
        kmers = [kmer, reverse_kmer]
        if not self._include_reverse_complement:
            kmers = [kmer]

        kmer_allele_frequency = min([self._graph.get_node_allele_frequency(node) for node in nodes])

        for kmer in kmers:

            self.results.append((kmer, start_node, start_offset, list(set(nodes))))

            if self._only_save_one_node_per_kmer:
                nodes = [nodes[0]]

            # add one hit for each unique node
            logging.info("Nodes: %s, unique: %s" % (nodes, np.unique(nodes)))
            for node in np.unique(nodes):
                logging.info("    Adding start node %d for node %d" % (start_node, node))
                self._start_nodes.append(start_node)
                self._start_offsets.append(start_offset)
                self._nodes.append(node)
                self._kmers.append(kmer)
                self._allele_frequencies.append(kmer_allele_frequency)

    def find(self):
        # iteratively calls search_from() on the various starting points (critical nodes)
        while self.search_from(self._current_search_start_node, self._current_search_start_offset, self._current_hash,
                                self._current_reverse_hash, 0):
            continue

        logging.info("N nodes skipped because too many variant nodes: %d" % self._n_nodes_skipped_because_too_complex)

    def _is_critical_node(self, node):
        if node in self._linear_ref_nodes:
            return True
        return False

    def _is_critical_position(self, node, offset):
        # critical position means that there are only one kmer ending at this position (not multiple paths before)
        # and the node is a critical node
        if self._critical_graph_paths.is_critical(node, offset):
            return True
        return False

    def search_from(self, node, offset, current_hash, current_reverse_hash, n_variant_nodes_passed):


        # change the current hash and current bases
        # if we don't have a long enough path, build hash incrementally by adding zero
        print(self._current_bases)
        first_base = self._current_bases[-self._k] if len(self._current_bases) >= self._k else 0
        first_base_complement = (first_base + 2) % 4 if len(self._current_bases) >= self._k else 0

        current_hash -= first_base * 4**(self._k-1)
        current_hash *= 4
        current_base = self._graph.get_numeric_base_sequence(node, offset)
        current_base_complement = (current_base+2) % 4
        current_hash += current_base

        current_reverse_hash -= first_base_complement
        current_reverse_hash /= 4
        current_reverse_hash += current_base_complement * 4**(self._k-1)

        self._current_bases.append(current_base)
        self._current_nodes.append(node)

        logging.info("On node %d, offset %d, current hash %d / %d, first base: %d" % (node, offset, current_hash, current_reverse_hash, first_base))


        if (node, offset, frozenset(self._current_nodes[-self._k:])) in self._positions_treated:
            logging.info("Already treated this exact position and path: %d/%d %s" % (node, offset, self._current_nodes[-self._k:]))
            return False

        self._positions_treated.add((node, offset, frozenset(self._current_nodes[-self._k:])))

        # starts a depth first search from this position until meeting a long enough critical node in graph
        if len(self._current_bases) >= self._k:
            self._add_kmer(current_hash, current_reverse_hash, node, offset)

        if not self._graph.is_linear_ref_node_or_linear_ref_dummy_node(node):
            n_variant_nodes_passed += 1


        # if at end of node, continue on edges
        if offset == self._graph.get_node_size(node)-1:
            next_nodes = self._graph.get_edges(node)
            if len(next_nodes) == 0:
                #logging.info("Stopping because no new nodes")
                return False

            if n_variant_nodes_passed > self._max_variant_nodes:
                # only allow next nodes on linear ref
                self._n_nodes_skipped_because_too_complex += len(next_nodes)
                next_nodes = [node for node in next_nodes if self._graph.is_linear_ref_node_or_linear_ref_dummy_node(node)]
                self._n_nodes_skipped_because_too_complex -= len(next_nodes)

            for i, next_node in enumerate(next_nodes):
                logging.info("")
                logging.info("  Continuing to node %d with bases %s" % (next_node, self._current_bases[-self._k:]))
                # processed
                n_bases_in_path = len(self._current_bases)
                result = self.search_from(next_node, 0, current_hash, current_reverse_hash, n_variant_nodes_passed)
                # after processing a chilld, reset current bases and nodes to where we are now before procesing next child
                logging.info("   Child is done, cutting path at %d" % n_bases_in_path)

                if i < len(next_nodes)-1:  # do not slice for last child, since a new recusion will start after that
                    self._current_bases.set_n_elements(n_bases_in_path)
                    self._current_nodes.set_n_elements(n_bases_in_path)

            return result
        else:

            if self._is_critical_position(node, offset+1):
                # stop recursion, next position is critical
                self._current_search_start_node = node
                self._current_search_start_offset = offset + 1
                self._current_hash = current_hash
                self._current_reverse_hash = current_reverse_hash
                logging.info("Stopping at critical position %d, %d, current hash %d, current bases %s\n"
                             % (self._current_search_start_node, self._current_search_start_offset, self._current_hash,
                                self._current_bases))
                return True
            return self.search_from(node, offset + 1, current_hash, current_reverse_hash,
                                    n_variant_nodes_passed)

