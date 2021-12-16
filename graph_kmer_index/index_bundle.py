import logging
import numpy as np
from obgraph.variant_to_nodes import VariantToNodes
from obgraph.numpy_variants import NumpyVariants
from graph_kmer_index import KmerIndex

# Wrapper for all indexes required by kage genotyping
class IndexBundle:
    index_names = ["VariantToNodes", "NumpyVariants", "NodeCountModelAdvanced", "HelperVariants", "CombinationMatrix", "TrickyVariants", "KmerIndex"]
    def __init__(self, indexes):
        self.indexes = indexes

    @classmethod
    def from_file(cls, file_name, skip=None):
        from kage.node_count_model import NodeCountModelAdvanced
        from kage.helper_index import HelperVariants, CombinationMatrix
        from kage.tricky_variants import TrickyVariants
        data = np.load(file_name)
        data_keys = list(data.keys())

        indexes = {}
        for index in cls.index_names:
            logging.info("Reading %s from index bundle" % index)
            if skip is not None and index in skip:
                logging.info("Not reading %s from index bundle" % index)
                continue
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
            logging.info("Done reading %s from index bundle" % index)

        return cls(indexes)

    def to_file(self, file_name, compress=True):
        from kage.node_count_model import NodeCountModelAdvanced
        from kage.helper_index import HelperVariants, CombinationMatrix
        from kage.tricky_variants import TrickyVariants
        archive = {}
        for index_name, object in self.indexes.items():
            logging.info("Packing index %s" % index_name)
            for property in eval(index_name).properties:
                archive_name = index_name + "." + property
                archive[archive_name] = getattr(object, property)

        if compress:
            logging.info("Saving to compressed file %s" % file_name)
            np.savez_compressed(file_name, **archive)
        else:
            logging.info("Saving to uncompressed file %s" % file_name)
            np.savez(file_name, **archive)


