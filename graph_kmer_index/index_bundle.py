import logging
import numpy as np
from obgraph.variant_to_nodes import VariantToNodes
from obgraph.numpy_variants import NumpyVariants
from graph_kmer_index import KmerIndex
from shared_memory_wrapper import from_file, to_file

# Wrapper for all indexes required by kage genotyping
class IndexBundle:
    index_names = ["VariantToNodes", "NumpyVariants", "NodeCountModelAdvanced", "HelperVariants", "CombinationMatrix", "TrickyVariants", "KmerIndex"]
    def __init__(self, indexes):
        self.indexes = indexes

    @classmethod
    def from_file(cls, file_name, skip=None):
        return cls(from_file(file_name))

    def to_file(self, file_name, compress=True):
        return to_file(self.indexes, file_name, compress=compress)
