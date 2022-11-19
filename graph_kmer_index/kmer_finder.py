import logging
import numpy as np
from .flat_kmers import FlatKmers2, FlatKmers
from .critical_graph_paths import CriticalGraphPaths
from .nplist import NpList
import sys
sys.setrecursionlimit(20000)
from obgraph.position_id import PositionId
from . import kmer_hash_to_sequence
from .kmer_hashing import power_array, reverse_power_array

_complement_lookup = np.array([3, 2, 1, 0], dtype=np.uint64)


def update_hash(current_base, current_hash, first_base, k, only_add=False):
    # only_add!=True means to build hash without subtracting previous first base
    # only_add should then be the number of the base we are at
    # very important that everything is int (not float) when working with large k
    current_hash = int(current_hash)
    #assert type(current_hash) == int, "Current hash has type %s" % type(current_hash)
    current_base = int(current_base)
    first_base = int(first_base)
    current_base_complement = _complement_lookup[current_base]  # (current_base + 2) % 4
    #print("Current base: %d. Only add: %s" % (current_base, only_add))
    if not isinstance(only_add, bool):
        #current_hash = current_hash * 4 + current_base
        current_hash = current_hash + 4**(only_add) * current_base
        #print("ONLY ADDING HASH. Current hash: %d, only add: %d" % (current_hash, only_add))
    else:
        #current_hash = (current_hash - first_base * 4 ** (k - 1)) * 4 + current_base
        current_hash = (current_hash - first_base) // 4 + current_base * 4**(k-1)
        first_base_complement = _complement_lookup[first_base]  # (first_base + 2) % 4

    return current_hash


class DenseKmerFinder:
    """
    Finds all possible kmers in graph
    """
    def __init__(self, graph, k, critical_graph_paths=None,
                 position_id=None, only_save_one_node_per_kmer=False, max_variant_nodes=4,
                 only_store_variant_nodes=False,
                 start_at_critical_path_number=None, stop_at_critical_path_number=None,
                 whitelist=None,
                 only_store_nodes=None,
                 only_follow_nodes=None):

        self._graph = graph
        self._linear_ref_nodes = self._graph.linear_ref_nodes()
        self._k = k
        self._only_save_one_node_per_kmer = only_save_one_node_per_kmer
        self._max_variant_nodes = max_variant_nodes
        self._start_nodes = NpList(dtype=np.int32)
        self._start_offsets = NpList(dtype=np.int16)
        self._nodes = NpList(dtype=np.int32)
        self._kmers = NpList(dtype=np.int64)
        self._allele_frequencies = NpList(dtype=float)
        
        self._power_vector = power_array(k)

        self._current_search_start_node = self._graph.get_first_node()
        self._current_search_start_offset = 0
        self._current_hash = 0
        self._current_bases = NpList(dtype=np.int8)
        self._current_nodes = NpList(dtype=np.int32)

        self.results = []

        self._n_nodes_skipped_because_too_complex = 0
        self._only_store_nodes = only_store_nodes
        self.kmers_found = []

        self._only_store_variant_nodes = only_store_variant_nodes
        if self._only_store_variant_nodes:
            assert variant_to_nodes is not None

        self._critical_graph_paths = critical_graph_paths

        self._position_id = position_id
        if position_id is None:
            logging.warning("Position id index is not set, creating")
            self._position_id = PositionId.from_graph(self._graph)

        self._nonempty_bases_traversed = 0

        self._effective_k = self._k  # how long we want to look back in path. Will be larger than x for empty nodes with dummy bases
        self._positions_treated = set()
        self._current_path_start_position = 0
        self._recursion_depth = 0
        
        self._stop_at_critical_path_number = stop_at_critical_path_number
        self._start_at_critical_path_number = start_at_critical_path_number

        self._whitelist = whitelist  # set of kmers, do not store any other than these
        self._n_skipped_whitelist = 0
        self._only_follow_nodes = only_follow_nodes

        self._early_stop = False

        self._whitelist = None
        if whitelist is not None:
            self._whitelist = whitelist
            #logging.info("Will limit kmers to whitelist (%d kmers in whitelist)" % len(whitelist))

    def get_found_kmers_and_nodes(self):
        return (self._kmers.get_nparray(), self._nodes.get_nparray())

    def get_flat_kmers(self, v="2"):
        assert self._allele_frequencies.get_nparray().dtype == float
        if v == "0" or v == "1":
            #logging.info("Converting start nodes/offsets to an iD to be compatible with FlatKmers")
            # return old version, convert start nodes and offsets to a position id
            start_nodes = self._start_nodes.get_nparray()
            start_offsets = self._start_offsets.get_nparray()
            if v == "1":
                ref_offsets = self._position_id.get(start_nodes, start_offsets)
            else:
                ref_offsets = self._graph.node_to_ref_offset[start_nodes]+start_offsets

            return FlatKmers(self._kmers.get_nparray(), self._nodes.get_nparray(), ref_offsets,
                             self._allele_frequencies.get_nparray())
        else:
            return FlatKmers2(self._kmers.get_nparray(), self._start_nodes.get_nparray(),
                          self._start_offsets.get_nparray(), self._nodes.get_nparray(),
                            self._allele_frequencies.get_nparray())

    def _add_kmer(self, kmer, start_node, start_offset):

        if self._whitelist is not None and kmer not in self._whitelist:
            self._n_skipped_whitelist += 1
            return

        nodes = np.unique(self._current_nodes[self._current_path_start_position:])
        #print("     Adding kmer %d/%s at node/offset %d/%d with nodes %s. Start node/offset: %d/%d" %
        #             (kmer, kmer_hash_to_sequence(kmer, self._k), start_node, start_offset, nodes, start_node, start_offset))

        n_variant_nodes = len([n for n in nodes if not self._graph.is_linear_ref_node_or_linear_ref_dummy_node(n)])
        if n_variant_nodes > self._max_variant_nodes:
            logging.warning("Passed too many variant nodes")


        kmer_allele_frequency = np.min(self._graph.get_node_allele_frequencies(nodes))

        if self._only_save_one_node_per_kmer:
            nodes = [nodes[0]]

        if len(self.kmers_found) < 500:
            nodes_added_set = set()
            
        # add one hit for each unique node
        for node in nodes:
            if self._only_store_nodes is not None and node not in self._only_store_nodes:
                continue

            #logging.info("    Adding start node %d for node %d" % (start_node, node))
            self._start_nodes.append(start_node)
            self._start_offsets.append(start_offset)
            self._nodes.append(node)
            self._kmers.append(kmer)
            self._allele_frequencies.append(kmer_allele_frequency)
            if len(self.kmers_found) < 500:
                nodes_added_set.add(node)

        if len(self.kmers_found) < 500:
            # Only add to this when there is little data, only used for testing and debugging
            # loggign.info("------Added kmer with nodes %s" % nodes)
            self.kmers_found.append((None, nodes_added_set, start_node, kmer))

    def find_only_kmers_starting_at_position(self, node, offset):
        # recuse all kmers from this position, always stop after a full kmer has been reached
        self._early_stop = True
        self._current_critical_node = node
        self._current_critical_offset = offset
        self._critical_graph_paths = CriticalGraphPaths.empty()
        #logging.info("Finding kmers starting at position %d/%d" % (node, offset))
        self.search_from(node, offset, 0)

    def find(self):
        if self._critical_graph_paths is None:
            logging.info("Making critical graph paths since it's not specified. "
                         "Will be faster if critical graph paths is premade")
            self._critical_graph_paths = CriticalGraphPaths.from_graph(self._graph, self._k)

        # iteratively calls search_from() on the various starting points (critical positions)

        logging.info("Stop at critical path number: %s" % self._stop_at_critical_path_number)
        logging.info("Start at critical path number: %s" % self._start_at_critical_path_number)

        self._starting_points = list(self._critical_graph_paths)[::-1]

        stop_at_node = None
        if self._stop_at_critical_path_number is not None and self._stop_at_critical_path_number < len(self._starting_points):
            stop_at_node = self._starting_points[-self._stop_at_critical_path_number-1][0]
            logging.info("Will stop at node %d" % stop_at_node)
        else:
            logging.info("Will stop at end of graph")


        self._starting_points_set = set()
        for starting_point in self._starting_points:
            self._starting_points_set.add(starting_point)

        if self._start_at_critical_path_number is not None and self._start_at_critical_path_number > 0:
            self._starting_points = self._starting_points[:-self._start_at_critical_path_number]  # remove the last

        # add beginning of graph as starting point if necessary
        if self._start_at_critical_path_number is None or self._start_at_critical_path_number == 0:
            if self._graph.get_node_size(self._graph.get_first_node()) <= self._k:  # means beginning is not a critical point, needs to add
                self._starting_points.append((self._graph.get_first_node(), 0))
                logging.info("Added first node of graph to critical nodes")


        #for critical_node, critical_offset in [(self._graph.get_first_node(), 0)] + list(self._critical_graph_paths):
        #logging.info("Starting points: %s" % self._starting_points)
        while len(self._starting_points) > 0:
            self._recursion_depth = 0
            critical_node, critical_offset = self._starting_points.pop()

            if stop_at_node is not None and stop_at_node == critical_node:
                logging.info("Stopping at critical path number %d" % self._stop_at_critical_path_number)
                break

            self._current_bases = NpList(dtype=np.int8)
            self._current_nodes = NpList()
            self._current_path_start_position = 0
            self._current_critical_node = critical_node
            self._current_critical_offset = critical_offset
            self._nonempty_bases_traversed = 0

            if critical_offset >= self._k-1:
                critical_offset -= (self._k-1)

            try:
                self.search_from(critical_node, critical_offset, 0)
            except RecursionError:
                logging.error("Failed searching from critical node %d and position %d" % (critical_node, critical_offset))
                logging.error("The graph might be too complex. Try setting max variants nodes lower?")
                logging.error("Recursion depth: %d" % self._recursion_depth)
                logging.error("Starting at critical path number: %d" % self._start_at_critical_path_number)
                raise

        logging.info("N nodes skipped because too many variant nodes: %d" % self._n_nodes_skipped_because_too_complex)
        logging.info("N skipped because whitelist: %d" % self._n_skipped_whitelist)


    def _is_critical_position(self, node, offset):
        # critical position means that there are only one kmer ending at this position (not multiple paths before)
        # and the node is a critical node
        if self._critical_graph_paths.is_critical(node, offset):
            return True
        return False

    def search_from(self, node, offset, current_hash):

        assert self._allele_frequencies._dtype == float, self._allele_frequencies._dtype

        self._recursion_depth += 1
        node_size = self._graph.get_node_size(node)

        if offset == 0 and self._graph.get_node_size(node) == 0:
            # this is a dummy node
            current_base = -1
            self._current_bases.append(current_base)
            self._current_nodes.append(node)


        while offset < node_size:

            # if we are in a middle of a big node, we can process this part more effectively since
            # we know there are no dummy nodes or edges or other stuff to think about
            if offset == self._k+2 and node_size > offset + self._k + 1 and not self._early_stop:
                current_hash, offset = self._process_whole_node(current_hash, node, node_size, offset)
            # change the current hash and current bases
            # if we don't have a long enough path, build hash incrementally by adding zero
            first_base = self._get_first_base_in_path()

            assert first_base != -1
            current_base = self._graph.get_numeric_base_sequence(node, offset)

            only_add = self._nonempty_bases_traversed  # len(self._current_bases)
            if current_base != -1:
                #if len(self._current_bases) >= self._k:
                if self._nonempty_bases_traversed >= self._k:
                    self._current_path_start_position += 1
                    only_add = False
                current_hash = update_hash(current_base, current_hash, first_base, self._k,
                                                                         only_add=only_add)
            if current_hash < 0:
                logging.error("Current node/offset: %d/%d" % (node, offset))
                logging.error("Current hash: %d" % current_hash)
                logging.error("First base: %d" % first_base)
                logging.error("Current base: %d" % current_base)
                raise Exception("Error in computing hash")

            self._current_bases.append(current_base)
            self._current_nodes.append(node)
            self._nonempty_bases_traversed += 1

            assert self._nonempty_bases_traversed <= len(self._current_bases)

            if False and ((node < 100 and offset < 15) or (offset > 0 and offset % 5000 == 0) or node % 5000 == 0):
                print("On node %d/%d, offset %d, %d kmers added. Skipped nodes: %d. "
                             "Path length: %d. Rec depth: %d. Nonempty traversed: %d. Nodes: %s"
                             % (node, len(self._graph.nodes), offset, len(self._kmers),
                                self._n_nodes_skipped_because_too_complex,
                                len(self._current_bases), self._recursion_depth, self._nonempty_bases_traversed,
                                ",".join((str(b) for b in (self._current_nodes[self._current_path_start_position:])))))
                #logging.info("Current search start node/offset: %d/%d" % (self._current_search_start_node, self._current_search_start_offset))

            current_path_desc = (node, offset, frozenset(self._current_nodes[self._current_path_start_position:]))
            if (node != self._current_critical_node or offset != self._current_critical_offset) and \
                    current_path_desc in self._positions_treated and len(self._current_nodes[self._current_path_start_position:]) >= self._k:
                self._recursion_depth -= 1
                #logging.info("Stopping recursion")
                #logging.info("current path desc: %s" % str((current_path_desc)))
                return False

            self._positions_treated.add(current_path_desc)

            # starts a depth first search from this position until meeting a long enough critical node in graph
            # do not add entries starting at empty nodes (we just want to include empty nodes in other entries)
            if self._nonempty_bases_traversed >= self._k and (current_base != -1 or self._early_stop):
                self._add_kmer(current_hash, node, offset)
                #logging.info("Added kmer")
                if self._early_stop:
                    #logging.info("Early stop")
                    # stop whenever a kmer is found
                    self._recursion_depth -= 1
                    return


            if (node != self._current_critical_node or offset + 1 != self._current_critical_offset) and \
                 self._is_critical_position(node, offset + 1):
                # stop recursion, next position is critical
                if (node, offset+1) not in self._starting_points_set:
                    self._starting_points.append((node, offset+1))
                    self._starting_points_set.add((node, offset+1))
                self._recursion_depth -= 1
                return False

            offset += 1

        assert offset == self._graph.get_node_size(node) or self._graph.get_node_size(node) == 0

        # at end of node, continue on edges
        self._search_next_nodes(current_hash, node)

    def _process_whole_node(self, current_hash, node, node_size, offset):
        sequence = self._graph.get_numeric_node_sequence(node)[offset - self._k:].astype(np.uint64)
        hashes = np.convolve(sequence, self._power_vector, mode='full')
        hashes = hashes[self._k:-self._k]  # get actual hashes after boundary effect and don't include last bp
        assert len(hashes) == node_size - offset - 1
        bases_to_extend = sequence[self._k:len(sequence) - 1]
        assert len(bases_to_extend) == len(hashes)
        self._current_bases.extend(bases_to_extend)
        self._current_nodes.extend(np.zeros(len(bases_to_extend)) + node)
        self._current_path_start_position += len(bases_to_extend)

        hashes_to_add = hashes
        offsets_to_add = np.arange(offset, node_size - 1)
        if self._whitelist is not None:
            hashes_to_add = np.array([h for h in hashes_to_add if h in self._whitelist])
            offsets_to_add = np.array([offset+i for i, h in enumerate(hashes) if h in self._whitelist])
            self._n_skipped_whitelist += (len(hashes)-len(hashes_to_add))

        n = len(hashes_to_add)


        self._kmers.extend(hashes_to_add)
        self._nodes.extend(np.zeros(n) + node)
        self._start_nodes.extend(np.zeros(n) + node)
        self._start_offsets.extend(offsets_to_add)
        self._allele_frequencies.extend(np.zeros(n, dtype=float) + self._graph.get_node_allele_frequency(node))

        # continue search from next offset and stop this search
        # NB: Converting to python int's to avoid problems when working with these hashes further
        current_hash = int(hashes[-1])
        # print("continuing with hash %d (type %s)" % (current_hash, type(current_hash)))
        offset = node_size - 1
        return current_hash, offset

    def _search_next_nodes(self, current_hash, node):
        next_nodes = self._graph.get_edges(node)
        force_follow = False
        if self._only_follow_nodes is not None and len(self._only_follow_nodes.intersection(next_nodes)) > 0:
            next_nodes = self._only_follow_nodes.intersection(next_nodes)
            force_follow = True

        if len(next_nodes) > 0:
            n_variant_nodes_passed = len(set([n for n in self._current_nodes[self._current_path_start_position:] if
                                              not self._graph.is_linear_ref_node_or_linear_ref_dummy_node(n)]))
            #logging.info("Searching next nodes from %d. Variant nodes until now: %d" % (node, n_variant_nodes_passed))
            if n_variant_nodes_passed > self._max_variant_nodes:
                logging.warning("Passed more variant nodes than planned. Could happen if forcing following certain nodes")

            if not force_follow and n_variant_nodes_passed >= self._max_variant_nodes:
                # only allow next nodes on linear ref
                self._n_nodes_skipped_because_too_complex += len(next_nodes)
                next_nodes = [node for node in next_nodes if
                              self._graph.is_linear_ref_node_or_linear_ref_dummy_node(node)]
                assert len(next_nodes) == 1, "Not 1 linear ref next nodes from node %d: %s" % (node, next_nodes)
                self._n_nodes_skipped_because_too_complex -= len(next_nodes)

            #logging.info("Next nodes: %s" % next_nodes)
            for i, next_node in enumerate(next_nodes):
                #logging.info("   Continuing on node %s. Nonempty bases: %d" % (next_node, self._nonempty_bases_traversed))
                path_start = self._current_path_start_position  # copy path start position before continuing recursion
                n_bases_in_path = len(self._current_bases)
                nonempty_bases_copy = self._nonempty_bases_traversed+0
                self.search_from(next_node, 0, current_hash)

                # after processing a child, reset current bases and nodes to where we are now before procesing next child
                self._current_bases.set_n_elements(n_bases_in_path)
                self._current_nodes.set_n_elements(n_bases_in_path)
                self._current_path_start_position = path_start
                self._nonempty_bases_traversed = nonempty_bases_copy

    def _get_first_base_in_path(self):
        #if len(self._current_bases) >= self._k:
        if self._nonempty_bases_traversed >= self._k:

            first_base = self._current_bases[self._current_path_start_position]

            # check if "dummy bases" are coming, then we want to skip them for the next iteration
            if len(self._current_bases) > self._current_path_start_position+1:
                next_first_base = self._current_bases[self._current_path_start_position + 1]
                while next_first_base == -1:
                    self._current_path_start_position += 1
                    next_first_base = self._current_bases[self._current_path_start_position + 1]

        else:
            first_base = 0
        return first_base
