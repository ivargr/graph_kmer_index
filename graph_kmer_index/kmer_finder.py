import logging


class DenseKmerFinder:
    """
    Finds all possible kmers in graph
    """
    def __init__(self, graph, k, add_reverse_complements=False):
        self._graph = graph
        self._linear_ref_nodes = self._graph.linear_ref_nodes()
        self._k = k
        self._add_reverse_complements = add_reverse_complements
        self._start_nodes = []
        self._start_offsets = []
        self._nodes = []
        self._kmers = []

        self._current_search_start_node = self._graph.get_first_node()
        self._current_search_start_offset = 0
        self._current_hash = 0
        self._current_bases = []

        self.results = []

    def _find_search_starting_points(self):
        # finds all critical nodes with length >= k in order to find suitable starting points for recursive searches
        pass

    def _add_kmer(self, kmer, start_node, start_offset):
        logging.info("Adding kmer %d at node/offset %d/%d" % (kmer, start_node, start_offset))
        self._kmers.append(kmer)
        self._start_nodes.append(start_node)
        self._start_offsets.append(start_offset)

        self.results.append((kmer, start_node, start_offset))

    def find(self):
        # iteratively calls search_from() on the various starting points (critical nodes)
        while self.search_from(self._current_search_start_node, self._current_search_start_offset, self._current_hash,
                               self._current_bases):
            continue

    def _is_critical_node(self, node):
        if node in self._linear_ref_nodes:
            return True

    def _is_critical_position(self, node, offset):
        # critical position means that there are only one kmer ending at this position (not multiple paths before)
        # and the node is a critical node
        if self._is_critical_node(node) and offset == self._k:
            return True
        return False

    def search_from(self, node, offset, current_hash, current_bases):

        # change the current hash and current bases
        first_base = current_bases.pop(0) if len(current_bases) == self._k else 0
        logging.info(" Subtracting first base %d*4**(k-1) = %d" % (first_base, first_base*4**(self._k-1)))
        current_hash -= first_base * 4**(self._k-1)
        current_hash *= 4
        current_base = self._graph.get_numeric_base_sequence(node, offset)
        current_hash += current_base
        current_bases.append(current_base)
        logging.info("On node %d, offset %d, current hash %d, current bases: %s" % (node, offset, current_hash, current_bases))

        # starts a depth first search from this position until meeting a long enough critical node in graph
        if len(current_bases) == self._k:
            self._add_kmer(current_hash, node, offset)

        # if at end of node, continue on edges
        if offset == self._graph.get_node_size(node)-1:
            next_nodes = self._graph.get_edges(node)
            if len(next_nodes) == 0:
                logging.info("Stopping because no new nodes")
                return False

            for next_node in next_nodes:
                logging.info("")
                logging.info("  Continuing to node %d with bases %s" % (next_node, current_bases.copy()))
                # todo: copy is slow, more effective to just slice the list after each node has been
                # processed
                result = self.search_from(next_node, 0, current_hash, current_bases.copy())

            return result
        else:

            if self._is_critical_position(node, offset):
                # stop recursion
                self._current_search_start_node = node
                self._current_search_start_offset = offset+1
                self._current_hash = current_hash
                self._current_bases = current_bases
                logging.info("Stopping at critical position %d, %d, current hash %d, current bases %s\n"
                             % (self._current_search_start_node, self._current_search_start_offset, self._current_hash, self._current_bases))
                return True

            return self.search_from(node, offset+1, current_hash, current_bases)

