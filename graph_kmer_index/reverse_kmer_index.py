import logging
import numpy as np


class ReverseKmerIndex:
    def __init__(self, nodes_to_index_positions, nodes_to_n_hashes, hashes):
        self.nodes_to_index_positions = nodes_to_index_positions
        self.nodes_to_n_hashes = nodes_to_n_hashes
        self.hashes = hashes

    def __str__(self):
        description = "Nodes to index positions: %s\n" % self.nodes_to_index_positions
        description += "Nodes to n hashes      : %s\n" % self.nodes_to_n_hashes
        description += "Hashes:                  %s\n" % self.hashes
        return description

    def __repr_(self):
        return self.__str__()

    def get_node_kmers(self, node):
        index_position = int(self.nodes_to_index_positions[node])
        n_hashes = int(self.nodes_to_n_hashes[node])
        if n_hashes == 0:
            return []

        return self.hashes[index_position:index_position+n_hashes]

    @classmethod
    def from_flat_kmers(cls, flat_kmers):
        logging.info("Creating ReverseKmerIndex from flat kmers")
        nodes = flat_kmers._nodes
        kmers = flat_kmers._hashes

        max_node = np.max(nodes)
        logging.info("Max node: %d" % max_node)

        nodes_index = np.zeros(max_node+1)
        n_kmers = np.zeros(max_node+1)
        sorted_nodes = np.argsort(nodes)
        print(sorted_nodes)
        nodes = nodes[sorted_nodes]
        kmers = kmers[sorted_nodes]


        diffs = np.ediff1d(nodes, to_begin=1)
        positions_of_unique_nodes = np.nonzero(diffs)[0]
        unique_nodes = nodes[positions_of_unique_nodes]

        nodes_index[unique_nodes] = positions_of_unique_nodes
        n_kmers_numbers = np.ediff1d(positions_of_unique_nodes, to_end=len(nodes)-positions_of_unique_nodes[-1])
        n_kmers[unique_nodes] = n_kmers_numbers
        return cls(nodes_index, n_kmers, kmers)


