from offsetbasedgraph import Graph, SequenceGraph, Block, Interval, NumpyIndexedInterval, Position
import numpy as np
from offsetbasedgraph.interval import NoLinearProjectionException
import logging
from collections import defaultdict
logging.basicConfig(level=logging.INFO)
from .flat_kmers import FlatKmers



class KmerFinder:
    def __init__(self, graph, sequence_graph, critical_nodes, linear_ref, k=3, chromosome=1, store_all_kmers=True):
        self.graph = graph
        self.sequence_graph = sequence_graph
        self.linear_ref = linear_ref
        self.linear_ref_nodes = linear_ref.nodes_in_interval()
        self._critical_nodes = critical_nodes
        self.k = k
        self.m = k
        self.w = 0
        self.chromosome = chromosome
        self._store_all_kmers = store_all_kmers

        self.max_search_to_node = self.graph.get_first_blocks()[0]
        #self.max_search_to_node = 117969570  # self.graph.get_first_blocks()[0]
        self._n_basepairs_traversed_on_critical_nodes = 0

        self.bases_in_path = []
        self.hashes_in_path = []
        self.nodes_in_path = []
        self.kmer_stored_in_path = []

        self.detected_minimizers = FlatKmers()

        self.visited_nodes = defaultdict(set)  # node id => list of last hashes for nodes, used to stop recursion
        self.unique_visited_nodes = set()

        self.recursion_depth = 0
        self.n_skipped_too_many_edges = 0
        self.n_skipped_visited_before = 0
        self.n_nodes_searched = 0
        self.visit_counter = defaultdict(int)
        
        self.max_graph_node = graph.max_block_id()
        self.print_debug("=================== STARTING ===============")

    def _get_last_bases_and_hashes_on_linear_ref(self, node, offset):
        # Gets the previous m bases and hashes
        ref_end = int(self.linear_ref.get_offset_at_position(Position(node, offset)))
        ref_start = int(self.linear_ref.get_offset_at_position(Position(node, 0)))
        #ref_start = ref_end - self.m
        #self.print_debug("Getting ref sequence between %d and %d" % (ref_start, ref_end))
        interval = self.linear_ref.get_exact_subinterval(ref_start, ref_end)
        bases = self.sequence_graph.get_interval_sequence(interval)
        bases = self.sequence_graph._letter_sequence_to_numeric(np.array(list(bases)))
        hashes = np.zeros(len(bases))
        kmer_stored_in_path = np.zeros(len(bases))
        bases_array = np.array(list(bases))
        # Get the last w hashes

        for i in range(self.k-1, len(bases_array)):
            sub_kmer = bases_array[i - self.k + 1:i+1]
            hashes[i] = kmer_to_hash_fast(sub_kmer, k=self.k)
            if i == self.k-1 or self._store_all_kmers or kmer_stored_in_path[i-self.k] == 1:
                self.detected_minimizers.add_kmer(hashes[i], node, i, [node], self.chromosome, ref_end)
                kmer_stored_in_path[i] = True
                #self.print_debug("Adding hash %s at ref position %d" % (hashes[i], i))

        return bases, hashes, np.zeros(len(hashes)) + node, kmer_stored_in_path

    def find_kmers(self):
        # We always start at end of max_search_to_node for simplicity (only issue is we don't search first node)
        # Fill last m hashes and bases
        i = 0
        while True:
            self._n_basepairs_traversed_on_critical_nodes = 0
            current_node = self.max_search_to_node
            #self.print_debug("New local search starting from node %d" % current_node)
            if current_node == self.max_graph_node:
                #self.print_debug("Doing nothing because last node")
                break

            #print("Starting new local search from node %d" % current_node)
            node_size = self.graph.blocks[current_node].length()
            next_nodes = self.graph.adj_list[current_node]
            if len(next_nodes) == 0:
                break

            if i == 0:
                bases_in_path, hashes_in_path, nodes_in_path, kmer_stored_in_path = self._get_last_bases_and_hashes_on_linear_ref(current_node, node_size)
                self.bases_in_path = list(bases_in_path)
                self.hashes_in_path = list(hashes_in_path)
                self.nodes_in_path = list(nodes_in_path)
                self.kmer_stored_in_path = list(kmer_stored_in_path)
            i += 1

            #self.print_debug("Bases in path: %s" % bases_in_path)
            #self.print_debug("Hashes in path: %s" % hashes_in_path)

            list_offset = len(self.bases_in_path)
            for j, next_node in enumerate(next_nodes):
                #self.print_debug("Starting local search from node %d. Bases in path now: %s" % (next_node, bases_in_path))
                self.recursion_depth += 1
                self._search_from_node(next_node)
                if j < len(next_nodes) - 1:
                    #self.print_debug("Slicing data at %d" % list_offset)
                    self.bases_in_path = self.bases_in_path[0:list_offset]
                    self.hashes_in_path = self.hashes_in_path[0:list_offset]
                    self.nodes_in_path = self.nodes_in_path[0:list_offset]
                    self.kmer_stored_in_path = self.kmer_stored_in_path[0:list_offset]

            if self.max_search_to_node == current_node:
                # We did not come any further, probably at end of graph
                break

        return self.detected_minimizers

    def _process_node(self, node_id):
        # For every base pair in node, dynamically calculate next hash
        node_base_values = self.sequence_graph.get_numeric_node_sequence(node_id)
        #self.print_debug("Node seq: %s" % self.sequence_graph.get_sequence(node_id))
        #self.print_debug("Hashes in path: %s" % self.hashes_in_path)
        #self.print_debug("Bases in path: %s" % self.bases_in_path)
        #self.print_debug("Nodes in path: %s" % self.nodes_in_path)
        #self.print_debug("Has stored in path: %s" % self.kmer_stored_in_path)
        for pos in range(0, self.graph.blocks[node_id].length()):
            prev_hash = self.hashes_in_path[-1]
            prev_base_value = self.bases_in_path[len(self.bases_in_path)-self.k]
            new_hash = prev_hash - pow(5, self.k-1) * prev_base_value
            new_hash *= 5
            new_hash = new_hash + node_base_values[pos]


            #try:
            #  linear_ref_pos, end = Interval(pos, pos+1, [node_id], self.graph).to_linear_offsets2(self.linear_ref)
            #except NoLinearProjectionException:
            #   logging.error("Could not project")
            #    linear_ref_pos = 0
            linear_ref_pos = 0

            self.nodes_in_path.append(node_id)
            self.bases_in_path.append(node_base_values[pos])
            self.hashes_in_path.append(new_hash)

            #self.detected_minimizers.add_minimizer(node_id, pos, new_hash, self.chromosome, linear_ref_pos)
            if self.kmer_stored_in_path[-self.k] or self._store_all_kmers:
                self.detected_minimizers.add_kmer(new_hash, node_id, pos, self.nodes_in_path[-self.k:], self.chromosome, linear_ref_pos)
                #self.print_debug("Adding hash %d at position %d" % (new_hash, pos))
                self.kmer_stored_in_path.append(True)
            else:
                self.kmer_stored_in_path.append(False)



    def print_debug(self, text):
        return
        print(' '.join("   " for _ in range(self.recursion_depth)) + text)

    def _search_from_node(self, node_id):

        #if node_id >= 555489:
        #    import sys
        #    sys.exit()
        
        self.n_nodes_searched += 1
        if node_id % 1000 == 0:
            logging.info("On node id %d. %d unique nodes. %d nodes searched, skipped too many edges: %d, "
                  "skipped on cache: %d,  path length %d, %d mm found, depth: %d, n times visited this: %d" %
                  (node_id, len(self.unique_visited_nodes), self.n_nodes_searched, self.n_skipped_too_many_edges, self.n_skipped_visited_before,
                   len(self.hashes_in_path), len(self.detected_minimizers.kmers), self.recursion_depth, self.visit_counter[node_id]))
        on_ref = False
        if node_id in self.linear_ref_nodes:
            on_ref = True
        #self.print_debug("")
        #self.print_debug("== Searching from node %d (depth: %d, on ref: %s, is crit: %s) == " % (node_id, self.recursion_depth, on_ref, node_id in self._critical_nodes))


        if node_id >= 117969598 + 10:
            import sys
            sys.exit()

        list_offset = len(self.bases_in_path) + 1
        assert len(self.bases_in_path) == len(self.hashes_in_path)

        if node_id in self._critical_nodes:
            self._n_basepairs_traversed_on_critical_nodes += self.graph.blocks[node_id].length()
        else:
            self._n_basepairs_traversed_on_critical_nodes = 0

        # Compute hashes for the rest of this node
        hash_of_last_w_hashes = sum(self.hashes_in_path[-self.k:])
        self.visited_nodes[node_id].add(hash_of_last_w_hashes)

        self.visit_counter[node_id] += 1
        self.unique_visited_nodes.add(node_id)
        self._process_node(node_id)

        hash_of_last_w_hashes = sum(self.hashes_in_path[-self.w:])

        # Start new
        if self._n_basepairs_traversed_on_critical_nodes > self.m:
            # Stop the recursion here
            self.max_search_to_node = max(node_id, self.max_search_to_node)
            #self.print_debug("   Stopping recursion (%d)" % self._n_basepairs_traversed_on_critical_nodes)
            self.recursion_depth -= 1
            return

        next_nodes = self.graph.adj_list[node_id]
        # Sort so that we prioritize reference nodes first
        next_nodes = sorted(next_nodes, reverse=True, key=lambda n: n in self.linear_ref_nodes)
        #self.print_debug("Possible next: %s" % next_nodes)
        for j, next_node in enumerate(next_nodes):

            # If recusion depth is high, we only continue to linear ref
            #self.print_debug("Prev hash to here: %s. Previous hashes to here: %s" % (hash_of_last_w_hashes, str(self.visited_nodes[next_node])))
            if hash_of_last_w_hashes in self.visited_nodes[next_node]:
                # Stop because we have visited this node before
                #self.print_debug("Skipping because visited before. Visisted %d times before" % self.visit_counter[node_id])
                self.n_skipped_visited_before += 1
            elif self.visit_counter[next_node] >= 10:
                #self.print_debug("Skipping next %d because visisted too many times."  % next_node)
                self.n_skipped_too_many_edges += 1

            #elif self.recursion_depth > 2 and next_node not in self.linear_ref_nodes and self.visit_counter[node_id] > 0:
            #    self.print_debug("Skipping next %d because recursion depth > 2 "  % next_node)
            #    self.n_skipped_too_many_edges += 1
            else:
                if len(next_nodes) > 1:
                    self.recursion_depth += 1
                self._search_from_node(next_node)
            # Slice the lists we used (cut away all we filled it with, so that the other
            # recursion start out with the lists as they were w
            # Do not slice on the last. On the last node, we are either done, or we are continuing search on next nodes again and want to keep
            if j < len(next_nodes) - 1:
                #self.print_debug("Slicing data from 0 to %d after processing search from node %d" % (list_offset, next_node))
                self.bases_in_path = self.bases_in_path[0:list_offset]
                self.hashes_in_path = self.hashes_in_path[0:list_offset]
                self.nodes_in_path = self.nodes_in_path[0:list_offset]
                self.kmer_stored_in_path = self.kmer_stored_in_path[0:list_offset]

        if len(next_nodes) > 1:
            self.recursion_depth -= 1



def test_simple1():
    graph = Graph({1: Block(10), 2: Block(1), 3: Block(1), 4: Block(10)}, {1: [2, 3], 2: [4], 3: [4]})
    graph.convert_to_numpy_backend()

    sequence_graph = SequenceGraph.create_empty_from_ob_graph(graph)
    sequence_graph.set_sequence(1, "GGGTTTATAC")
    sequence_graph.set_sequence(2, "A")
    sequence_graph.set_sequence(3, "C")
    sequence_graph.set_sequence(4, "GTACATTGTA")

    linear_ref = Interval(0, 10, [1, 2, 3], graph)
    linear_ref = linear_ref.to_numpy_indexed_interval()

    critical_nodes = set([4])

    finder = KmerFinder(graph, sequence_graph, critical_nodes, linear_ref, k=3)
    kmers = finder.find_kmers()
    kmers.assert_has_kmer("GGG", 1, 2, [1])
    kmers.assert_has_kmer("GGT", 1, 3, [1])
    kmers.assert_has_kmer("ACA", 2, 0, [1, 2])
    kmers.assert_has_kmer("CCG", 4, 0, [1, 3, 4])


    # Do not store everything
    finder = KmerFinder(graph, sequence_graph, critical_nodes, linear_ref, k=3, store_all_kmers=False)
    kmers = finder.find_kmers()
    print("Kmers found: ")
    print(kmers.kmers)
    kmers.assert_has_kmer("GGG", 1, 2, [1])
    kmers.assert_not_has_kmer("GGT", 1, 3, [1])
    kmers.assert_has_kmer("TTT", 1, 5, [1])
    kmers.assert_has_kmer("ATA", 1, 8, [1])
    kmers.assert_has_kmer("CCG", 4, 0, [1, 3, 4])
    kmers.assert_has_kmer("CAG", 4, 0, [1, 2, 4])
    kmers.assert_has_kmer("TAC", 4, 3, [4])


def test_simple2():
    graph = Graph({1: Block(10), 2: Block(1), 3: Block(1), 4: Block(10), 5: Block(2), 6: Block(1), 7: Block(8), 8: Block(1)},
                  {1: [2, 3], 2: [4], 3: [4], 4: [5, 6], 5: [7], 6: [7], 7: [8]})
    graph.convert_to_numpy_backend()

    sequence_graph = SequenceGraph.create_empty_from_ob_graph(graph)
    sequence_graph.set_sequence(1, "GGGTTTATAC")
    sequence_graph.set_sequence(2, "A")
    sequence_graph.set_sequence(3, "C")
    sequence_graph.set_sequence(4, "GTACATTGTA")
    sequence_graph.set_sequence(5, "GG")
    sequence_graph.set_sequence(6, "A")
    sequence_graph.set_sequence(7, "AGGGGAAA")
    sequence_graph.set_sequence(8, "A")

    linear_ref = Interval(0, 8, [1, 2, 3, 4, 6, 7, 8], graph)
    linear_ref = linear_ref.to_numpy_indexed_interval()

    critical_nodes = set([4, 7])

    finder = KmerFinder(graph, sequence_graph, critical_nodes, linear_ref, k=4, store_all_kmers=True)
    kmers = finder.find_kmers()
    kmers.assert_has_kmer("TAG", 5, 0, [4, 5])
    kmers.assert_has_kmer("AAA", 7, 0, [4, 6, 7])

    # larger k
    finder = KmerFinder(graph, sequence_graph, critical_nodes, linear_ref, k=5)
    kmers = finder.find_kmers()
    kmers.assert_has_kmer("TAGGA", 7, 0, [4, 5, 7])
    kmers.assert_has_kmer("AAAGG", 7, 2, [4, 6, 7])


def test_simple2_sparse_kmers():
    graph = Graph({1: Block(10), 2: Block(1), 3: Block(1), 4: Block(10), 5: Block(2), 6: Block(1), 7: Block(8), 8: Block(1)},
                  {1: [2, 3], 2: [4], 3: [4], 4: [5, 6], 5: [7], 6: [7], 7: [8]})
    graph.convert_to_numpy_backend()

    sequence_graph = SequenceGraph.create_empty_from_ob_graph(graph)
    sequence_graph.set_sequence(1, "GGGTTTATAC")
    sequence_graph.set_sequence(2, "A")
    sequence_graph.set_sequence(3, "C")
    sequence_graph.set_sequence(4, "GTACATTGTA")
    sequence_graph.set_sequence(5, "GG")
    sequence_graph.set_sequence(6, "A")
    sequence_graph.set_sequence(7, "AGGGGAAA")
    sequence_graph.set_sequence(8, "A")

    linear_ref = Interval(0, 8, [1, 2, 3, 4, 6, 7, 8], graph)
    linear_ref = linear_ref.to_numpy_indexed_interval()

    critical_nodes = set([4, 7])

    finder = KmerFinder(graph, sequence_graph, critical_nodes, linear_ref, k=4, store_all_kmers=False)
    kmers = finder.find_kmers()
    kmers.assert_has_kmer("ACCG", 4, 0, [1, 3, 4])
    kmers.assert_has_kmer("ACAG", 4, 0, [1, 2, 4])
    kmers.assert_has_kmer("TACA", 4, 4, [4])
    kmers.assert_has_kmer("AGGA", 7, 0, [4, 5, 7])
    kmers.assert_not_has_kmer("GTAC", 4, 3, [4])
    kmers.assert_has_kmer("AAAG", 7, 1, [4, 6, 7])
    kmers.assert_has_kmer("GGGG", 7, 4, [7])




def test_many_nodes_sparse_kmers():
    nodes = {i: Block(1) for i in range(2, 10)}
    nodes[1] = Block(10)
    nodes[10] = Block(10)

    graph = Graph(nodes,
                  {1: [2, 3],
                   2: [4],
                   3: [4],
                   4: [5, 6],
                   5: [7],
                   6: [7],
                   7: [8, 9],
                   8: [10],
                   9: [10]})

    graph.convert_to_numpy_backend()
    sequence_graph = SequenceGraph.create_empty_from_ob_graph(graph)
    sequence_graph.set_sequence(1, "ACTGACTGAC")
    sequence_graph.set_sequence(10, "ACTGACTGAC")
    sequence_graph.set_sequence(2, "A")
    sequence_graph.set_sequence(3, "C")
    sequence_graph.set_sequence(4, "A")
    sequence_graph.set_sequence(5, "G")
    sequence_graph.set_sequence(6, "C")
    sequence_graph.set_sequence(7, "T")
    sequence_graph.set_sequence(8, "A")
    sequence_graph.set_sequence(9, "A")

    linear_ref = Interval(0, 10, [1, 2, 4, 6, 7, 8, 10], graph)
    linear_ref = linear_ref.to_numpy_indexed_interval()
    critical_nodes = {1, 4, 7, 10}

    finder = KmerFinder(graph, sequence_graph, critical_nodes, linear_ref, k=4, store_all_kmers=False)
    kmers = finder.find_kmers()
    kmers.assert_has_kmer("ACTG", 1, 3, [1])
    kmers.assert_has_kmer("ACAA", 4, 0, [1, 2, 4])
    kmers.assert_has_kmer("ACCA", 4, 0, [1, 3, 4])
    kmers.assert_has_kmer("GTAA", 10, 0, [5, 7, 8, 10])
    kmers.assert_has_kmer("CTAA", 10, 0, [6, 7, 8, 10])
    kmers.assert_has_kmer("CTGA", 10, 4, [10])
    kmers.assert_has_kmer("CTGA", 10, 8, [10])
    kmers.assert_not_has_kmer("TAAC", 10, 1, [7, 8, 10])


def test_many_nodes():
    nodes = {i: Block(1) for i in range(2, 10)}
    nodes[1] = Block(10)
    nodes[10] = Block(10)

    graph = Graph(nodes,
                  {1: [2, 3],
                   2: [4],
                   3: [4],
                   4: [5, 6],
                   5: [7],
                   6: [7],
                   7: [8, 9],
                   8: [10],
                   9: [10]})

    graph.convert_to_numpy_backend()
    sequence_graph = SequenceGraph.create_empty_from_ob_graph(graph)
    sequence_graph.set_sequence(1, "ACTGACTGAC")
    sequence_graph.set_sequence(10, "ACTGACTGAC")
    sequence_graph.set_sequence(2, "A")
    sequence_graph.set_sequence(3, "C")
    sequence_graph.set_sequence(4, "A")
    sequence_graph.set_sequence(5, "G")
    sequence_graph.set_sequence(6, "C")
    sequence_graph.set_sequence(7, "T")
    sequence_graph.set_sequence(8, "A")
    sequence_graph.set_sequence(9, "A")

    linear_ref = Interval(0, 10, [1, 2, 4, 6, 7, 8, 10], graph)
    linear_ref = linear_ref.to_numpy_indexed_interval()
    critical_nodes = {1, 4, 7, 10}

    finder = KmerFinder(graph, sequence_graph, critical_nodes, linear_ref, k=3)


if __name__ == "__main__":
    # WOrking tests
    test_simple1()
    test_many_nodes_sparse_kmers()
    test_simple2_sparse_kmers()

