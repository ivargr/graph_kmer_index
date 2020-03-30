import logging
import numpy as np
from collections import defaultdict


class FlatKmers:
    def __init__(self, hashes, nodes, ref_offsets):
        self._hashes = hashes
        self._nodes = nodes
        self._ref_offsets = ref_offsets

    @classmethod
    def from_file(cls, file_name):
        data = np.load(file_name)
        logging.info("Loaded kmers from %s" % file_name)
        return cls(data["hashes"], data["nodes"], data["ref_offsets"])

    def to_file(self, file_name):
        np.savez(file_name, hashes=self._hashes, nodes=self._nodes, ref_offsets=self._ref_offsets)
        logging.info("Save dto %s.npz" % file_name)


def letter_sequence_to_numeric(sequence):
    if not isinstance(sequence, np.ndarray):
        sequence = np.array(list(sequence.lower()))

    numeric = np.zeros_like(sequence, dtype=np.int64)
    numeric[np.where(sequence == "n")[0]] = 0
    numeric[np.where(sequence == "a")[0]] = 0
    numeric[np.where(sequence == "c")[0]] = 1
    numeric[np.where(sequence == "t")[0]] = 2
    numeric[np.where(sequence == "g")[0]] = 3
    numeric[np.where(sequence == "m")[0]] = 0
    return numeric


