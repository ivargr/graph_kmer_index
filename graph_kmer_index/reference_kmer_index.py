import logging
import numpy as np


class ReferenKmerIndex:
    def __init__(self, ref_position_to_index, kmers):
        self.ref_position_to_index = ref_position_to_index
        self.kmers = kmers

    def get_between(self, ref_start, ref_end):
        return self.kmers[
            self.ref_position_to_index[ref_start]:self.ref_position_to_index[ref_end]
        ]

    @classmethod
    def from_flat_kmers(cls, flat_kmers):
        ref_positions = flat_kmers._ref_offsets
        sorting = np.argsort(ref_positions)
        ref_positions = ref_positions[sorting]
        kmers = flat_kmers._hashes[sorting]

        positions_of_new_ref_positions = np.where(np.ediff1d(ref_positions, to_begin=0))[0]
        ref_position_to_index = np.zeros(int(ref_positions[-1]) + 1, dtype=np.uint64)

        ref_position_to_index[ref_positions[positions_of_new_ref_positions]] = positions_of_new_ref_positions

        return cls(ref_position_to_index, kmers)

    def to_file(self, file_name):
        np.savez(file_name,
                 ref_position_to_index=self.ref_position_to_index,
                 kmers=self.kmers)

    @classmethod
    def from_file(cls, file_name):
        try:
            data = np.load(file_name + ".npz")
        except FileNotFoundError:
            data = np.load(file_name)

        return cls(data["ref_position_to_index"], data["kmers"])