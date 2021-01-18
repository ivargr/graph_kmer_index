import numpy as np

class VariantNodesIndex:
    def __init__(self, ref_positions, variant_nodes):
        self.ref_position = ref_positions
        self.variant_nodes = variant_nodes

    @classmethod
    def from_file(cls, file_name):
        data = np.load(file_name)
        return cls(data["ref_positions"], data["variant_nodes"])

    def get_variant_nodes_between_ref_positions(self, ref_start, ref_end):


    @classmethod
    def from_graph(cls, graph):
        pass

    def to_file(self, file_name):
        np.savez(file_name, ref_position=self.ref_position, variant_nodes=self.variant_nodes)
