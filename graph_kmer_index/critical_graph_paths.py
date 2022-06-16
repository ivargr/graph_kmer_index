import numpy as np
import logging


class CriticalGraphPaths:
    def __init__(self, nodes, offsets, index=None):
        self.nodes = nodes
        self.offsets = offsets
        self._index = index

    def _make_index(self):
        if len(self.nodes) == 0:
            self._index = np.zeros(0)
            return

        self._index = np.zeros(np.max(self.nodes)+1, dtype=np.uint16)
        self._index[self.nodes] = self.offsets

    @classmethod
    def empty(cls):
        return cls(np.array([]), np.array([]), np.array([]))

    def is_critical(self, node, offset):
        if self._index is None:
            logging.info("Making critical paths index")
            logging.info("NOdes: %s, offsets: %s" % (self.nodes, self.offsets))
            self._make_index()

        if node >= len(self._index):
            return False

        return self._index[node] == offset

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return ((node, offset) for node, offset in zip(self.nodes, self.offsets))

    @classmethod
    def from_graph(cls, graph, k):

        logging.info("Getting reverse edges")
        reverse_edges = graph.get_reverse_edges_hashtable()

        critical_nodes = []
        critical_offsets = []


        logging.info("Iterating chromosome start nodes %s" % graph.chromosome_start_nodes)
        for start_node in graph.chromosome_start_nodes:

            #logging.info("On chromosome start node %d" % start_node)
            current_node = start_node
            depth = 0
            bp_since_last_join = 0
            i = 0
            while True:
                if i % 100000 == 0:
                    logging.info("%d / %d nodes traversed. %d critical paths found." % (i, graph.max_node_id(), len(critical_nodes)))
                i += 1

                # lower depth with number of edges coming in
                prev_depth = depth
                depth -= len(reverse_edges[current_node])
                if prev_depth > 1 and depth == 0:
                    #print("Resetting bp since last join, Prev depth: %d" % prev_depth)
                    bp_since_last_join = 0

                #print("Node %d, depth %d. Bp since last join: %d" % (current_node, depth, bp_since_last_join))

                node_size = graph.get_node_size(current_node)

                if depth == 0 and node_size != 0:  # ignore dummy nodes
                    #if bp_since_last_join + node_size >= k:
                    if bp_since_last_join <= k and bp_since_last_join + node_size >= k:
                        # we have a critical path point somewhere on this node
                        critical_nodes.append(current_node)
                        #print("  adding node %d offset %d. Bp since last join is now %d" % (current_node, k-bp_since_last_join-1, bp_since_last_join))
                        critical_offsets.append(k - bp_since_last_join-1)
                        #bp_since_last_join += node_size

                next_nodes = graph.get_edges(current_node)
                depth += len(next_nodes)

                if len(next_nodes) == 0:
                    break
                elif len(next_nodes) == 1:
                    bp_since_last_join += node_size
                    #print(" single next node, increased bp since last join to %d (node size of %d is %d) " % (bp_since_last_join, current_node, node_size))
                    current_node = next_nodes[0]
                else:
                    next_nodes = [n for n in next_nodes if graph.is_linear_ref_node_or_linear_ref_dummy_node(n)]
                    if len(next_nodes) != 1:
                        logging.error("Did not find 1 next node from node %d" % current_node)
                        logging.error("Edges: %s" % graph.get_edges(current_node))
                        raise Exception("")
                    current_node = next_nodes[0]

        logging.info("Found %d critical paths" % len(critical_nodes))
        return cls(np.array(critical_nodes, dtype=np.uint32), np.array(critical_offsets, np.uint16))


