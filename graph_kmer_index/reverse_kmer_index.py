import logging
import numpy as np


class ReverseKmerIndex:
    properties = {"nodes_to_index_positions", "nodes_to_n_hashes", "hashes", "ref_positions"}
    def __init__(self, nodes_to_index_positions=None, nodes_to_n_hashes=None, hashes=None, ref_positions=None):
        self.nodes_to_index_positions = nodes_to_index_positions
        self.nodes_to_n_hashes = nodes_to_n_hashes
        self.hashes = hashes
        self.ref_positions = ref_positions

    def __str__(self):
        description = "Nodes to index positions: %s\n" % self.nodes_to_index_positions
        description += "Nodes to n hashes      : %s\n" % self.nodes_to_n_hashes
        description += "Hashes:                  %s\n" % self.hashes
        description += "Ref positions:                  %s\n" % self.ref_positions
        return description

    def __repr_(self):
        return self.__str__()

    def get_node_kmers(self, node):
        index_position = int(self.nodes_to_index_positions[node])
        n_hashes = int(self.nodes_to_n_hashes[node])
        if n_hashes == 0:
            return []

        return self.hashes[index_position:index_position+n_hashes]

    def get_node_kmers_and_ref_positions(self, node):
        try:
            index_position = int(self.nodes_to_index_positions[node])
        except IndexError:
            logging.error("Invalid node %d" % node)
            raise

        n_hashes = int(self.nodes_to_n_hashes[node])
        if n_hashes == 0:
            return [[], []]

        return self.hashes[index_position:index_position+n_hashes], self.ref_positions[index_position:index_position+n_hashes]

    @classmethod
    def from_file(cls, file_name):
        try:
            data = np.load(file_name)
        except FileNotFoundError:
            data = np.load(file_name + ".npz")

        return cls(data["nodes_to_index_positions"], data["nodes_to_n_hashes"], data["hashes"], data["ref_positions"])

    def to_file(self, file_name):
        np.savez(file_name, nodes_to_index_positions=self.nodes_to_index_positions,
                            nodes_to_n_hashes=self.nodes_to_n_hashes,
                            hashes=self.hashes,
                            ref_positions=self.ref_positions)

    @classmethod
    def from_flat_kmers(cls, flat_kmers):
        logging.info("Creating ReverseKmerIndex from flat kmers")
        nodes = flat_kmers._nodes
        kmers = flat_kmers._hashes
        ref_positions = flat_kmers._ref_offsets

        max_node = np.max(nodes)
        logging.info("Max node: %d" % max_node)

        nodes_index = np.zeros(max_node+1, dtype=np.uint32)
        n_kmers = np.zeros(max_node+1, dtype=np.uint16)
        sorted_nodes = np.argsort(nodes)
        nodes = nodes[sorted_nodes]
        kmers = kmers[sorted_nodes]
        ref_positions = ref_positions[sorted_nodes]


        diffs = np.ediff1d(nodes, to_begin=1)
        positions_of_unique_nodes = np.nonzero(diffs)[0]
        unique_nodes = nodes[positions_of_unique_nodes]

        nodes_index[unique_nodes] = positions_of_unique_nodes
        n_kmers_numbers = np.ediff1d(positions_of_unique_nodes, to_end=len(nodes)-positions_of_unique_nodes[-1])
        n_kmers[unique_nodes] = n_kmers_numbers
        return cls(nodes_index, n_kmers, kmers, ref_positions)


