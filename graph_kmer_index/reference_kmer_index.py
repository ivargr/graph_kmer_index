import logging
import numpy as np
from pyfaidx import Fasta
from .read_kmers import ReadKmers
from .kmer_hashing import power_array

"""
@numba.jit
def fill_zeros_from_end(array):
    for i in range(1, len(array)):
        index_pos = len(array) - i
        if array[index_pos] == 0:
            array[index_pos] = array[index_pos+1]
"""

def fill_zeros_from_end(array):
    array = array[::-1]
    prev = np.arange(len(array))
    prev[array == 0] = 0
    prev = np.maximum.accumulate(prev)
    return array[prev][::-1]


class ReferenceKmerIndex:
    properties = {"ref_position_to_index", "kmers", "ref_positions", "nodes"}
    def __init__(self, ref_position_to_index=None, kmers=None, ref_positions=None, nodes=None):
        self.ref_position_to_index = ref_position_to_index
        self.kmers = kmers
        self.ref_positions = ref_positions
        self.nodes = nodes

    def get_between(self, ref_start, ref_end):
        return self.kmers[
            self.ref_position_to_index[ref_start]:self.ref_position_to_index[min(len(self.ref_position_to_index)-1, ref_end)]
        ]

    def get_between_except(self, ref_start, ref_end, except_position):
        assert self.ref_positions is None
        indexes = [i for i in np.arange(ref_start, ref_end) if i != except_position]
        return self.kmers[indexes]

    def get_all_between(self, ref_start, ref_end):
        if self.ref_positions is None:
            raise Exception("This index is missing reference positions and cannot be used to get all between. "
                            "Is it made from a linear reference? If so, use get_between() instead")
        start = self.ref_position_to_index[ref_start]
        end = self.ref_position_to_index[ref_end]
        return self.kmers[start:end], self.ref_positions[start:end], self.nodes[start:end]

    @classmethod
    def from_sequence(cls, genome_sequence, k, only_store_kmers=False):
        kmers = ReadKmers.get_kmers_from_read_dynamic(genome_sequence, power_array(k))

        ref_position_to_index = None
        if not only_store_kmers:
            ref_position_to_index = np.arange(0, len(genome_sequence), dtype=np.uint32)
        else:
            logging.info("Only storing kmers, not index")

        if k <= 16:
            kmers = kmers.astype(np.uint32)
            logging.info("Converting kmers to 32 bit uint, since k is small enough.")
        else:
            logging.info("Kmers are stored using 64 bits")
            kmers = kmers.astype(np.uint64)

        return cls(ref_position_to_index, kmers)

    @classmethod
    def from_linear_reference(cls, fasta_file_name, reference_name="ref", k=15, only_store_kmers=False):
        logging.info("Only store kmers? %s" % only_store_kmers)
        logging.info("k=%d" % k)
        genome_sequence = str(Fasta(fasta_file_name)[reference_name])
        return cls.from_sequence(genome_sequence, k, only_store_kmers)

    @classmethod
    def from_flat_kmers(cls, flat_kmers):
        ref_positions = flat_kmers._ref_offsets
        sorting = np.argsort(ref_positions)
        ref_positions = ref_positions[sorting]
        kmers = flat_kmers._hashes[sorting]
        logging.info("Checking if kmers can be stored as 32 bit")
        if np.max(kmers) < 2**32:
            logging.warning("Storing kmers as 32 bit uint since max hash is low enough")
            kmers = kmers.astype(np.uint32)

        nodes = flat_kmers._nodes[sorting]

        assert len(kmers) < 4294967295, "Too many kmers to store (32 bit limit reached). There are %d kmers" % len(kmers)

        positions_of_new_ref_positions = np.where(np.ediff1d(ref_positions, to_begin=0))[0]
        ref_position_to_index = np.zeros(int(ref_positions[-1]) + 1, dtype=np.uint32)

        ref_position_to_index[ref_positions[positions_of_new_ref_positions]] = positions_of_new_ref_positions

        # Fill zeros in ref_position_to_index
        # if there are not kmers at every ref position, there will be some zeros (ref positions mapping nowhere)
        # we want this to map to the next ref position that has a kmer
        """
        n_corrected = 0
        for i in range(1, len(ref_position_to_index)):
            if i % 1000000 == 0:
                logging.info("Filling zeros, %d processed" % i)
            index_pos = len(ref_position_to_index) - i
            if ref_position_to_index[index_pos] == 0:
                n_corrected += 1
                ref_position_to_index[index_pos] = ref_position_to_index[index_pos+1]

        logging.info("Filled %d zeros" % n_corrected)
        """
        logging.info("Filling zeroes")
        ref_position_to_index = fill_zeros_from_end(ref_position_to_index)

        return cls(ref_position_to_index, kmers, ref_positions, nodes)

    def to_file(self, file_name):

        if self.ref_position_to_index is None:
            np.savez(file_name, kmers=self.kmers)
        elif self.ref_positions is None and self.nodes is None:
            np.savez(file_name,
                 ref_position_to_index=self.ref_position_to_index,
                 kmers=self.kmers)
        else:
            np.savez(file_name,
                 ref_position_to_index=self.ref_position_to_index,
                 kmers=self.kmers,
                 ref_positions=self.ref_positions,
                 nodes=self.nodes)

    @classmethod
    def from_file(cls, file_name):
        try:
            data = np.load(file_name + ".npz")
        except FileNotFoundError:
            data = np.load(file_name)

        nodes = None
        ref_positions = None
        ref_position_to_index = None

        if "nodes" in data:
            nodes = data["nodes"]
        if "ref_positions" in data:
            ref_positions = data["ref_positions"]
        if "ref_position_to_index" in data:
            ref_position_to_index = data["ref_position_to_index"]

        return cls(ref_position_to_index, data["kmers"], ref_positions, nodes)