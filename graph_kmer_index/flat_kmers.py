import logging
import numpy as np
from collections import defaultdict
from graph_minimap.find_minimizers_in_kmers import kmer_to_hash_fast

CHAR_VALUES = {"a": 0, "g": 1, "c": 2, "t": 3, "n": 4, "A": 0, "G": 1, "C": 2, "T": 3, "N": 4}
CHAR_VALUES_STR = {"a": "0", "g": "1", "c": "2", "t": "3", "n": "4", "A": "0", "G": "1", "C": "2", "T": "3",
                       "N": "4"}



class FlatKmers:
    def __init__(self, hashes, nodes):
        self._hashes = hashes
        self._nodes = nodes

    @classmethod
    def from_file(cls, file_name):
        data = np.load(file_name)
        logging.info("Loaded kmers from %s" % file_name)
        return cls(data["hashes"], data["nodes"])

    def to_file(self, file_name):
        np.savez(file_name, hashes=self._hashes, nodes=self._nodes)
        logging.info("Save dto %s.npz" % file_name)


def letter_sequence_to_numeric(sequence):
    if not isinstance(sequence, np.ndarray):
        sequence = np.array(list(sequence.lower()))

    numeric = np.zeros_like(sequence, dtype=np.uint8)
    numeric[np.where(sequence == "n")[0]] = 0
    numeric[np.where(sequence == "a")[0]] = 1
    numeric[np.where(sequence == "c")[0]] = 2
    numeric[np.where(sequence == "t")[0]] = 3
    numeric[np.where(sequence == "g")[0]] = 4
    numeric[np.where(sequence == "m")[0]] = 4
    return numeric


