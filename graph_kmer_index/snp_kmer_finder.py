import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from offsetbasedgraph import Graph, Interval, Block, SequenceGraph
from graph_minimap.find_minimizers_in_kmers import kmer_to_hash_fast
from .flat_kmers import FlatKmers, letter_sequence_to_numeric



class SnpKmerFinder:
    """
    Simple kmer finder that only supports SNP graphs
    """

    def __init__(self, graph, sequence_graph, linear_ref, k=15):
        self.graph = graph
        self.sequence_graph = sequence_graph
        self.linear_ref = linear_ref
        self.k = k
        self.linear_nodes = linear_ref.nodes_in_interval()
        self._hashes = []
        self._nodes = []
        self.kmers_found = []
        self._bases_in_search_path = []
        self._nodes_in_path = []
        self._kmers_found = 0


    def has_kmer(self, kmer, nodes):
        if (kmer, nodes) in self.kmers_found:
            return True
        return False


    def _add_kmer(self, kmer, nodes):
        #logging.info("Adding kmer %s, %s" % (kmer, nodes))
        self._kmers_found += 1
        if len(self.kmers_found) < 200:
            # Only add to this when there is little data, only used for testing
            self.kmers_found.append((kmer, nodes))

        hash = kmer_to_hash_fast(letter_sequence_to_numeric(kmer), k=len(kmer))
        for node in nodes:
            self._hashes.append(hash)
            self._nodes.append(node)

    def _find_all_variant_kmers_from_position(self, linear_ref_pos):

        # Always start one base pair before, but do not include that base pair
        # this lets us catch cases where we are at a beginning of a node

        if linear_ref_pos > 0:
            node = self.linear_ref.get_node_at_offset(linear_ref_pos-1)
            offset = self.linear_ref.get_node_offset_at_offset(linear_ref_pos-1) + 1
        else:
            node = self.linear_ref.get_node_at_offset(linear_ref_pos)
            offset = self.linear_ref.get_node_offset_at_offset(linear_ref_pos)

        self._bases_in_search_path = []
        self._nodes_in_path = []
        self._search_graph_from(node, offset, self.k)


    def _search_graph_from(self, node, offset, bases_left):
        #logging.info("== Searching node %d from offset %d. Bases left: %d == " % (node, offset, bases_left))

        if bases_left == 0:
            self._add_kmer(''.join(self._bases_in_search_path), set(self._nodes_in_path))
            #logging.info("Recursion is done")
            return

        # Process the rest of this node
        node_size = int(self.graph.blocks[node].length())
        node_sequence = self.sequence_graph.get_node_sequence(node)
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
        next_nodes = self.graph.adj_list[node]
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
        for i in range(0, self.linear_ref.length() // self.k):
            pos = i * self.k
            if i % 10000 == 0:
                logging.info("On ref position %d. %d kmers found" % (pos, self._kmers_found))
            #if pos > 3000000:
            #break
            self._find_kmers_from_linear_ref_position(pos)

        logging.info("Done finding all kmers")
        return FlatKmers(np.array(self._hashes, dtype=np.uint64), np.array(self._nodes, np.uint32))



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
