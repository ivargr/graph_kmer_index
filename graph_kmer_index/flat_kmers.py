import logging
import numpy as np
from collections import defaultdict


class FlatKmers:
    def __init__(self, hashes, nodes, ref_offsets=None):
        self._hashes = hashes
        self._nodes = nodes
        if ref_offsets is None:
            self._ref_offsets = np.zeros(len(self._nodes))
        else:
            self._ref_offsets = ref_offsets

    @classmethod
    def from_file(cls, file_name):
        try:
            data = np.load(file_name)
        except FileNotFoundError:
            data = np.load(file_name + ".npz")

        logging.info("Loaded kmers from %s" % file_name)
        return cls(data["hashes"], data["nodes"], data["ref_offsets"])

    def to_file(self, file_name):
        np.savez(file_name, hashes=self._hashes, nodes=self._nodes, ref_offsets=self._ref_offsets)
        logging.info("Save dto %s.npz" % file_name)


    @classmethod
    def from_multiple_flat_kmers(cls, flat_kmers_list):
        hashes = []
        nodes = []
        ref_offsets = []
        for flat in flat_kmers_list:
            hashes.extend(flat._hashes)
            nodes.extend(flat._nodes)
            if flat._ref_offsets is not None:
                ref_offsets.extend(flat._ref_offsets)

        if len(ref_offsets) == 0:
            ref_offsets = None
        else:
            ref_offsets = np.array(ref_offsets, np.uint64)

        return FlatKmers(np.array(hashes, dtype=np.uint64), np.array(nodes, np.uint32), ref_offsets)

def letter_sequence_to_numeric(sequence):
    if not isinstance(sequence, np.ndarray):
        sequence = np.array(list(sequence.lower()), dtype="<U1")

    numeric = np.zeros_like(sequence, dtype=np.int64)
    numeric[np.where(sequence == "n")[0]] = 0
    numeric[np.where(sequence == "a")[0]] = 0
    numeric[np.where(sequence == "c")[0]] = 1
    numeric[np.where(sequence == "t")[0]] = 2
    numeric[np.where(sequence == "g")[0]] = 3
    numeric[np.where(sequence == "m")[0]] = 0
    return numeric


