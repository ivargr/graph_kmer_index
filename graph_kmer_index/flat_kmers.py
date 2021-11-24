import logging
import numpy as np
from collections import defaultdict


class FlatKmers:
    def __init__(self, hashes, nodes, ref_offsets=None, allele_frequencies=None):
        assert len(hashes) == len(nodes)
        self._hashes = hashes
        self._nodes = nodes
        if ref_offsets is None:
            self._ref_offsets = np.zeros(len(self._nodes))
        else:
            self._ref_offsets = ref_offsets

        if allele_frequencies is None:
            logging.info("Allele frequencies not provided. Setting all to 1.0")
            self._allele_frequencies = np.zeros(len(self._hashes), dtype=np.single) + 1.0
        else:
            self._allele_frequencies = allele_frequencies


    @classmethod
    def from_file(cls, file_name):
        try:
            data = np.load(file_name)
        except FileNotFoundError:
            data = np.load(file_name + ".npz")

        logging.info("Loaded kmers from %s" % file_name)
        return cls(data["hashes"], data["nodes"], data["ref_offsets"], data["allele_frequencies"])

    def to_file(self, file_name):
        np.savez(file_name, hashes=self._hashes, nodes=self._nodes, ref_offsets=self._ref_offsets, allele_frequencies=self._allele_frequencies)
        logging.info("Save dto %s.npz" % file_name)


    @classmethod
    def from_multiple_flat_kmers(cls, flat_kmers_list):
        logging.info("Making flat kmers")
        hashes = []
        nodes = []
        ref_offsets = []
        allele_frequencies = []
        for flat in flat_kmers_list:
            hashes.extend(flat._hashes)
            nodes.extend(flat._nodes)
            if flat._ref_offsets is not None:
                ref_offsets.extend(flat._ref_offsets)

            allele_frequencies.extend(flat._allele_frequencies)

        if len(ref_offsets) == 0:
            ref_offsets = None
        else:
            ref_offsets = np.array(ref_offsets, np.uint64)

        logging.info("Done making flat kmers")
        return FlatKmers(np.array(hashes, dtype=np.uint64), np.array(nodes, np.uint32), ref_offsets, np.array(allele_frequencies, dtype=np.uint16))

    def sum_of_kmer_frequencies(self, kmer_index_with_frequencies):
        return sum([0] + [max(1, kmer_index_with_frequencies.get_frequency(int(kmer))) for kmer in self._hashes])

    def maximum_kmer_frequency(self, kmer_index_with_frequencies):
        return max([0] + [kmer_index_with_frequencies.get_frequency(int(kmer)) for kmer in self._hashes])
    
    def get_new_without_singletons(self):
        has_been_traversed = set()
        new_hashes = []
        new_nodes = []
        new_ref_offsets = []
        new_allele_frequencies = []

        for i in range(len(self._hashes)):
            if i % 1000000 == 0:
                logging.info("%d entries processed. %d kept" % (i, len(new_hashes)))

            hash = self._hashes[i]
            if hash in has_been_traversed:
                new_hashes.append(hash)
                new_nodes.append(self._nodes[i])
                new_ref_offsets.append(self._ref_offsets[i])
                new_allele_frequencies.append(self._allele_frequencies[i])
            else:
                has_been_traversed.add(hash)

        logging.info("Making new numpy arrays")
        new_hashes = np.array(new_hashes, dtype=self._hashes.dtype)
        new_nodes = np.array(new_nodes, dtype=self._nodes.dtype)
        new_ref_offsets = np.array(new_ref_offsets, dtype=self._ref_offsets.dtype)
        new_allele_frequencies = np.array(new_allele_frequencies, dtype=self._allele_frequencies.dtype)

        logging.info("Returning new flat kmers")
        return FlatKmers(new_hashes, new_nodes, new_ref_offsets, new_allele_frequencies)


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

def numeric_to_letter_sequence(sequence):

    numeric = np.zeros_like(sequence, dtype=object)
    numeric[np.where(sequence == 0)[0]] = "a"
    numeric[np.where(sequence == 1)[0]] = "c"
    numeric[np.where(sequence == 2)[0]] = "t"
    numeric[np.where(sequence == 3)[0]] = "g"
    return numeric





