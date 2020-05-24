import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from offsetbasedgraph import Graph, Interval, Block, SequenceGraph
from .flat_kmers import FlatKmers, letter_sequence_to_numeric
from numba import jit
from Bio.Seq import Seq
from collections import defaultdict


def sequence_to_kmer_hash(sequence):
    return kmer_to_hash_fast(letter_sequence_to_numeric(sequence), len(sequence))

@jit(nopython=True)
def kmer_to_hash_fast(kmer, k):
    numbers = int(np.sum(kmer * np.power(4, np.arange(0, k)[::-1])))
    return numbers


class SnpKmerFinder:
    """
    Simple kmer finder that only supports SNP graphs
    """

    def __init__(self, graph, k=15, spacing=None, include_reverse_complements=False, pruning=False, max_kmers_same_position=100000,
                 max_frequency=10000, max_variant_nodes=10000):
        self.graph = graph
        self.k = k
        self.linear_nodes = graph.linear_ref_nodes()
        self._hashes = []
        self._nodes = []
        self._ref_offsets = []
        self.kmers_found = []
        self._bases_in_search_path = []
        self._nodes_in_path = []
        self._kmers_found = 0
        self._current_ref_offset = None
        self._last_ref_pos_added = 0
        self.pruning = pruning
        self._n_kmers_pruned = 0
        self._has_traversed_variant = False
        self._unique_kmers_added = set()
        self._max_kmers_same_position = max_kmers_same_position
        self._n_kmers_added_current_position = 0
        self._n_kmers_skipped = 0
        self._kmer_frequencies = defaultdict(int)
        self._max_frequency = max_frequency
        self._n_skipped_due_to_frequency = 0
        self._max_variant_nodes = max_variant_nodes
        self._n_skipped_due_to_max_variant_nodes = 0


        if self.pruning:
            logging.info("Will do pruning")
        else:
            logging.info("No pruning will be performed, all kmers will be reported")

        if spacing is None:
            self.spacing = k
        else:
            self.spacing = spacing
        self._include_reverse_complements=include_reverse_complements
        if self._include_reverse_complements:
            logging.info("Will include reverse complement of kmers")


    def has_kmer(self, kmer, nodes):
        if (kmer, nodes) in self.kmers_found:
            return True
        return False


    def _add_kmer(self, kmer, nodes):
        #logging.info("Adding kmer %s, %s" % (kmer, nodes))

        hash = kmer_to_hash_fast(letter_sequence_to_numeric(kmer), k=len(kmer))

        if self._kmer_frequencies[hash] > self._max_frequency:
            self._n_skipped_due_to_frequency += 1
            return

        if self.pruning and hash not in self._unique_kmers_added:
            if self._last_ref_pos_added != self._current_ref_offset and self._last_ref_pos_added > self._current_ref_offset - 124:
                if not self._has_traversed_variant:
                    # Do not add
                    self._n_kmers_pruned += 1
                    return

        if self._n_kmers_added_current_position > self._max_kmers_same_position:
            self._n_kmers_skipped += 1
            return

        n_variant_nodes = len([n for n in nodes if n not in self.linear_nodes])
        if n_variant_nodes > self._max_variant_nodes:
            self._n_skipped_due_to_max_variant_nodes += 1
            return

        if self._include_reverse_complements:
            rev_hash = kmer_to_hash_fast(letter_sequence_to_numeric(str(Seq(kmer).reverse_complement())), k=len(kmer))

        self._unique_kmers_added.add(hash)
        self._kmer_frequencies[hash] += 1

        for node in nodes:
            self._hashes.append(hash)
            self._nodes.append(node)
            self._ref_offsets.append(self._current_ref_offset)

            if self._include_reverse_complements:
                self._hashes.append(rev_hash)
                self._nodes.append(node)
                self._ref_offsets.append(self._current_ref_offset)

        self._last_ref_pos_added = self._current_ref_offset

        self._kmers_found += 1
        if len(self.kmers_found) < 200:
            # Only add to this when there is little data, only used for testing
            self.kmers_found.append((kmer, nodes))

        self._n_kmers_added_current_position += 1

    def _find_all_variant_kmers_from_position(self, linear_ref_pos):
        self._current_ref_offset = linear_ref_pos

        # Always start one base pair before, but do not include that base pair
        # this lets us catch cases where we are at a beginning of a node

        if linear_ref_pos > 0:
            node = self.graph.get_node_at_ref_offset(linear_ref_pos-1)
            offset = self.graph.get_node_offset_at_ref_offset(linear_ref_pos-1) + 1
        else:
            node = self.graph.get_node_at_ref_offset(linear_ref_pos)
            offset = self.graph.get_node_offset_at_ref_offset(linear_ref_pos)

        self._bases_in_search_path = []
        self._nodes_in_path = []
        self._has_traversed_variant = False
        self._n_kmers_added_current_position = 0
        self._search_graph_from(node, offset, self.k)



    def _search_graph_from(self, node, offset, bases_left):
        #logging.info("== Searching node %d from offset %d. Bases left: %d == " % (node, offset, bases_left))

        if bases_left == 0:
            self._add_kmer(''.join(self._bases_in_search_path), set(self._nodes_in_path))
            #logging.info("Recursion is done")
            return

        # Process the rest of this node
        node_size = int(self.graph.nodes[node])
        node_sequence = self.graph.get_node_sequence(node)
        for node_position in range(int(offset), node_size):
            base = node_sequence[node_position]
            #logging.info("   Adding base %s at node %d, pos %d" % (base, node, node_position))
            self._bases_in_search_path.append(base)
            self._nodes_in_path.append(node)
            bases_left -= 1

            if bases_left == 0:
                self._add_kmer(''.join(self._bases_in_search_path), set(self._nodes_in_path))
                #logging.info("Recursion is done")
                # Recursion is done
                return

        # If offset is last base in this node, recursively search next nodes depth first
        next_nodes = self.graph.get_edges(node)
        if len(next_nodes) > 1:
            self._has_traversed_variant = True

        # Prioritize linear ref node
        if len(next_nodes) > 0 and next_nodes[0] not in self.linear_nodes:
            next_nodes = list(reversed(next_nodes))

        bases_so_far = len(self._bases_in_search_path)
        for next_node in next_nodes:
            # After a search, reset the bases in search path back to where it was
            self._search_graph_from(next_node, 0, bases_left)
            #logging.info("Limiting at %d"  % bases_so_far)
            self._bases_in_search_path = self._bases_in_search_path[0:bases_so_far]
            self._nodes_in_path = self._nodes_in_path[0:bases_so_far]


    def _find_kmers_from_linear_ref_position(self, pos):
        self._find_all_variant_kmers_from_position(pos)


    def find_kmers(self):
        for i in range(0, self.graph.linear_ref_length() // self.spacing):
            pos = i * self.spacing
            if i % 10000 == 0:
                logging.info("On ref position %d. %d kmers found. Have pruned %d kmers. "
                             "Skipped %d kmers. Skipped due to high frequency: %d. Skipped because too many variant nodes: %d"
                             % (pos, self._kmers_found, self._n_kmers_pruned, self._n_kmers_skipped, self._n_skipped_due_to_frequency, self._n_skipped_due_to_max_variant_nodes))
            #if pos > 3000000:
            #break
            self._find_kmers_from_linear_ref_position(pos)

        logging.info("Done finding all kmers")
        return FlatKmers(np.array(self._hashes, dtype=np.uint64), np.array(self._nodes, np.uint32), np.array(self._ref_offsets, np.uint64))



def test_simple():

    graph = Graph({1: Block(10), 2: Block(1), 3: Block(1), 4: Block(10)}, {1: [2, 3], 2: [4], 3: [4]})
    graph.convert_to_numpy_backend()

    sequence_graph = SequenceGraph.create_empty_from_ob_graph(graph)
    sequence_graph.set_sequence(1, "GGGTTTATAC")
    sequence_graph.set_sequence(2, "A")
    sequence_graph.set_sequence(3, "C")
    sequence_graph.set_sequence(4, "GTACATTGTA")

    linear_ref = Interval(0, 10, [1, 3, 4], graph)
    linear_ref = linear_ref.to_numpy_indexed_interval()

    finder = SnpKmerFinder(graph, sequence_graph, linear_ref, k=5)
    finder.find_kmers()
    assert finder.has_kmer("gggtt", {1})
    assert finder.has_kmer("cgtac", {3, 4})
    assert finder.has_kmer("agtac", {2, 4})
    assert finder.has_kmer("agtac", {2, 4})
    assert finder.has_kmer("attgt", {4})

if __name__ == "__main__":

    test_simple()
