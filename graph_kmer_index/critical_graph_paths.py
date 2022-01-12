import numpy as np
import logging


class CriticalGraphPaths:
    def __init__(self, nodes, offsets, index=None):
        self.nodes = nodes
        self.offsets = offsets
        self._index = index

    def _make_index(self):
        self._index = np.zeros(np.max(self.nodes)+1, dtype=np.uint16)
        self._index[self.nodes] = self.offsets

    def is_critical(self, node, offset):
        if self._index is None:
            logging.info("Making critical paths index")
            self._make_index()

        if node >= len(self._index):
            return False

        return self._index[node] == offset

    @classmethod
    def from_graph(cls, graph, k):

        reverse_edges = graph.get_reverse_edges_hashtable()

        critical_nodes = []
        critical_offsets = []

        current_node = graph.get_first_node()
        depth = 0
        bp_since_last_join = 0
        while True:
            # lower depth with number of edges coming in
            prev_depth = depth
            depth -= len(reverse_edges[current_node])
            if prev_depth > 1 and depth == 0:
                print("Resetting bp since last join, Prev depth: %d" % prev_depth)
                bp_since_last_join = 0

            print(reverse_edges[current_node])
            print("Node %d, depth %d. Bp since last join: %d" % (current_node, depth, bp_since_last_join))

            node_size = graph.get_node_size(current_node)

            if depth == 0 and node_size != 0:  # ignore dummy nodes
                if bp_since_last_join + node_size >= k:
                    # we have a critical path point somewhere on this node
                    critical_nodes.append(current_node)
                    print("  adding offset %d" % (k-bp_since_last_join-1))
                    critical_offsets.append(k - bp_since_last_join-1)
                    bp_since_last_join += node_size

            next_nodes = graph.get_edges(current_node)
            depth += len(next_nodes)

            if len(next_nodes) == 0:
                break
            elif len(next_nodes) == 1:
                bp_since_last_join += node_size
                print(" single next node, increased bp since last join to %d" % bp_since_last_join)
                current_node = next_nodes[0]
            else:
                next_nodes = [n for n in next_nodes if graph.is_linear_ref_node_or_linear_ref_dummy_node(n)]
                assert len(next_nodes) == 1
                current_node = next_nodes[0]

        return cls(np.array(critical_nodes, dtype=np.uint32), np.array(critical_offsets, np.uint16))


