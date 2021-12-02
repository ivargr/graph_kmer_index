import logging
import numpy as np
from obgraph.variant_to_nodes import VariantToNodes
from obgraph.numpy_variants import NumpyVariants
from .node_count_model import NodeCountModelAdvanced
from .helper_index import HelperVariants, CombinationMatrix
from .tricky_variants import TrickyVariants
from graph_kmer_index import KmerIndex

# Wrapper for all indexes required by kage genotyping
class IndexBundle:
    index_names = ["VariantToNodes", "NumpyVariants", "NodeCountModelAdvanced", "HelperVariants", "CombinationMatrix", "TrickyVariants", "KmerIndex"]
    def __init__(self, indexes):
        self.indexes = indexes

    @classmethod
    def from_file(cls, file_name):
        data = np.load(file_name)
        data_keys = list(data.keys())

        indexes = {}
        for index in cls.index_names:
            property_data = {}
            properties = eval(index).properties
            for property in properties:
                try:
                    property_data[property] = data[index + "." + property]
                except KeyError:
                    logging.error("Tried to get index %s with property %s that is not in the bundle" % (index, property))
                    logging.error("Available in the bundle: %s" % data_keys)
                    raise

            index_object = eval(index)(**property_data)
            indexes[index] = index_object

        return cls(indexes)

    def to_file(self, file_name):
        archive = {}
        for index_name, object in self.indexes.items():
            for property in eval(index_name).properties:
                archive_name = index_name + "." + property
                archive[archive_name] = getattr(object, property)

        np.savez_compressed(file_name, **archive)


