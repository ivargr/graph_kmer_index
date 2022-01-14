import logging
import numpy as np
from .flat_kmers import FlatKmers2
from .critical_graph_paths import CriticalGraphPaths
from npstructures.numpylist import NumpyList
import sys
#sys.setrecursionlimit(100000)

class DenseKmerFinder:
    """
    Finds all possible kmers in graph
    """
    def __init__(self, graph, k, critical_graph_paths=None, only_save_one_node_per_kmer=False, max_variant_nodes=4,
                 include_reverse_complements=False, variant_to_nodes=None, only_store_variant_nodes=False):
        self._graph = graph
        self._linear_ref_nodes = self._graph.linear_ref_nodes()
        self._k = k
        self._only_save_one_node_per_kmer = only_save_one_node_per_kmer
        self._max_variant_nodes = max_variant_nodes
        self._start_nodes = NumpyList(dtype=np.int32)
        self._start_offsets = NumpyList(dtype=np.int16)
        self._nodes = NumpyList(dtype=np.int32)
        self._kmers = NumpyList(dtype=np.int64)
        self._allele_frequencies = NumpyList(np.single)

        self._current_search_start_node = self._graph.get_first_node()
        self._current_search_start_offset = 0
        self._current_hash = 0
        self._current_reverse_hash = 0
        self._current_bases = NumpyList(dtype=np.int8)
        self._current_nodes = NumpyList()

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

        self._effective_k = self._k  # how long we want to look back in path. Will be larger than x for empty nodes with dummy bases
        self._positions_treated = set()
        self._current_path_start_position = 0
        self._recursion_depth = 0

    def get_flat_kmers(self):
        return FlatKmers2(self._kmers.get_nparray(), self._start_nodes.get_nparray(),
                          self._start_offsets.get_nparray(), self._nodes.get_nparray(),
                            self._allele_frequencies.get_nparray())

    def _find_search_starting_points(self):
        # finds all critical nodes with length >= k in order to find suitable starting points for recursive searches
        pass

    def _add_kmer(self, kmer, reverse_kmer, start_node, start_offset):
        nodes = self._current_nodes[self._current_path_start_position:]
        #logging.info("     Adding kmer %d at node/offset %d/%d with nodes %s. Start node/offset: %d/%d" %
        #             (kmer, start_node, start_offset, nodes, start_node, start_offset))
        kmers = [kmer, reverse_kmer]
        if not self._include_reverse_complement:
            kmers = [kmer]

        kmer_allele_frequency = min([self._graph.get_node_allele_frequency(node) for node in nodes])

        for kmer in kmers:

            #self.results.append((kmer, start_node, start_offset, list(set(nodes))))

            if self._only_save_one_node_per_kmer:
                nodes = [nodes[0]]

            # add one hit for each unique node
            #logging.info("Nodes: %s, unique: %s" % (nodes, np.unique(nodes)))
            for node in np.unique(nodes):
                #logging.info("    Adding start node %d for node %d" % (start_node, node))
                self._start_nodes.append(start_node)
                self._start_offsets.append(start_offset)
                self._nodes.append(node)
                self._kmers.append(kmer)
                self._allele_frequencies.append(kmer_allele_frequency)

    def find(self):
        # iteratively calls search_from() on the various starting points (critical positions)
        print(self._critical_graph_paths.nodes, self._critical_graph_paths.offsets)


        self._starting_points = list(self._critical_graph_paths)[::-1]
        if self._graph.get_node_size(self._graph.get_first_node()) < self._k:
            self._starting_points.append((self._graph.get_first_node(), 0))

        self._starting_points_set = set()
        for starting_point in self._starting_points:
            self._starting_points_set.add(starting_point)

        #print("Starting points: %s" % self._starting_points)

        #for critical_node, critical_offset in [(self._graph.get_first_node(), 0)] + list(self._critical_graph_paths):
        while len(self._starting_points) > 0:
            self._recursion_depth = 0
            critical_node, critical_offset = self._starting_points.pop()

            #logging.info("Searching recursively from node/offset %d/%d" % (critical_node, critical_offset))
            self._current_bases = NumpyList(dtype=np.int8)
            self._current_nodes = NumpyList()
            self._current_path_start_position = 0

            self._current_critical_node = critical_node
            self._current_critical_offset = critical_offset

            if critical_offset >= self._k-1:
                critical_offset -= (self._k-1)
                #logging.info("  Adjusted critical offset to %d" % critical_offset)
                # we are starting into a node, build nodes, bases and hash before starting to add entries
                #self._current_nodes = NumpyList(dtype=np.uint8)
                #self._current_nodes.extend([critical_node]*self._k)
                #self._current_bases.extend([])
            res = self.search_from(critical_node, critical_offset, 0, 0)


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

    def search_from(self, node, offset, current_hash, current_reverse_hash):
        self._recursion_depth += 1



        if offset == 0 and self._graph.get_node_size(node) == 0:
            # this is a dummy node
            current_base = -1
            self._current_bases.append(current_base)
            self._current_nodes.append(node)

        for offset in range(offset, self._graph.get_node_size(node)):
            # change the current hash and current bases
            # if we don't have a long enough path, build hash incrementally by adding zero
            #print("----- node/offset: %d/%d. Path start pos: %d" % (node, offset, self._current_path_start_position))
            if len(self._current_bases) >= self._k:

                first_base = self._current_bases[self._current_path_start_position]

                # check if "dummy bases" are coming, then we want to skip them for the next iteration
                next_first_base = self._current_bases[self._current_path_start_position+1]
                while next_first_base == -1:
                    self._current_path_start_position += 1
                    next_first_base = self._current_bases[self._current_path_start_position+1]

                # next first base will be -1, we don't want to include this in our path
                #if self._current_bases[self._current_path_start_position+1] == -1:
                #    self._current_path_start_position += 1
                #    #print("FIRST BASE in path is now a dummy base. Increas path start position with 1 to %d" % self._current_path_start_position)
            else:
                first_base = 0

            #if len(self._current_bases) >= self._k:
            #    if self._current_bases[-self._effective_k+1] == -1:
            #        self._effective_k -= 1

            assert first_base != -1
            first_base_complement = (first_base + 2) % 4 if len(self._current_bases) >= self._k else 0

            current_base = self._graph.get_numeric_base_sequence(node, offset)
            #print("BASE at %d/%d: %d" % (node, offset, current_base))

            if current_base != -1:
                if len(self._current_bases) >= self._k:
                    self._current_path_start_position += 1

                current_hash = (current_hash - first_base * 4 ** (self._k - 1)) * 4 + current_base
                current_base_complement = (current_base+2) % 4
                current_reverse_hash -= (current_reverse_hash-first_base_complement) / 4 + \
                                        current_base_complement * 4**(self._k-1)

            assert current_hash >= 0

            self._current_bases.append(current_base)
            self._current_nodes.append(node)

            if (node < 100 and offset < 10) or (offset > 0 and offset % 10000 == 0) or node % 10000 == 0:
                #logging.info("On node %d, offset %d, current hash %d / %d, first base: %d. Path length: %d. Current bases: %s. Nodes. %s. Path start pos: %d. path start offset: %d"
                #         % (node, offset, current_hash, current_reverse_hash, first_base, len(self._current_bases),
                #            self._current_bases[-35:], self._current_nodes[-35:], self._current_path_start_position, len(self._current_bases)-self._current_path_start_position))

                logging.info("On node %d/%d, offset %d, %d kmers added. Skipped nodes: %d. current hash %d / %d, first base: %d. Path length: %d. Recusion depth: %d"
                             % (node, len(self._graph.nodes), offset, len(self._kmers), self._n_nodes_skipped_because_too_complex, current_hash, current_reverse_hash, first_base, len(self._current_bases), self._recursion_depth))
                #logging.info("Current search start node/offset: %d/%d" % (self._current_search_start_node, self._current_search_start_offset))

            current_path_desc = (node, offset, frozenset(self._current_nodes[self._current_path_start_position:]))
            if (node != self._current_critical_node or offset != self._current_critical_offset) and \
                    current_path_desc in self._positions_treated and len(self._current_nodes) >= self._k:
                #logging.info("!!!!! Already treated this exact position and path: %s" % str(current_path_desc))
                return True

            self._positions_treated.add(current_path_desc)

            # starts a depth first search from this position until meeting a long enough critical node in graph
            # do not add entries starting at empty nodes (we just want to include empty nodes in other entries)
            if len(self._current_bases) >= self._k and current_base != -1:
                self._add_kmer(current_hash, current_reverse_hash, node, offset)


            if (node != self._current_critical_node or offset + 1 != self._current_critical_offset) and \
                 self._is_critical_position(node, offset + 1):
                # stop recursion, next position is critical
                if (node, offset+1) not in self._starting_points_set:
                    self._starting_points.append((node, offset+1))
                    self._starting_points_set.add((node, offset+1))
                #else:
                #    logging.info("Did not add starting point %d/%d since added before" % (node, offset+1))

                #logging.info("Stopping at critical position %d, %d\n" % (node, offset))
                return True

        assert offset == self._graph.get_node_size(node)-1 or self._graph.get_node_size(node) == 0

        # if at end of node, continue on edges
        if offset == self._graph.get_node_size(node)-1 or self._graph.get_node_size(node) == 0:
            next_nodes = self._graph.get_edges(node)
            if len(next_nodes) == 0:
                #logging.info("Stopping because no new nodes")
                return False

            n_variant_nodes_passed = len(set([n for n in self._current_nodes[self._current_path_start_position:] if not self._graph.is_linear_ref_node_or_linear_ref_dummy_node(n)]))
            logging.info("N variant nodes passed when at end of node %d: %d" % (node, n_variant_nodes_passed))
            if n_variant_nodes_passed >= self._max_variant_nodes:
                # only allow next nodes on linear ref
                self._n_nodes_skipped_because_too_complex += len(next_nodes)
                next_nodes = [node for node in next_nodes if self._graph.is_linear_ref_node_or_linear_ref_dummy_node(node)]
                assert len(next_nodes) == 1
                self._n_nodes_skipped_because_too_complex -= len(next_nodes)

            for i, next_node in enumerate(next_nodes):
                #logging.info("")
                #logging.info("  Continuing to node %d with bases %s" % (next_node, self._current_bases[-self._k:]))
                # processed
                path_start = self._current_path_start_position
                n_bases_in_path = len(self._current_bases)
                result = self.search_from(next_node, 0, current_hash, current_reverse_hash)
                # after processing a chilld, reset current bases and nodes to where we are now before procesing next child

                self._current_bases.set_n_elements(n_bases_in_path)
                self._current_nodes.set_n_elements(n_bases_in_path)
                self._current_path_start_position = path_start

            return result


        else:
            return self.search_from(node, offset + 1, current_hash, current_reverse_hash)

