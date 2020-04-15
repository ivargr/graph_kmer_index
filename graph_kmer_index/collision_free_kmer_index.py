import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from .logn_hash_map import LogNHashMap, ModuloHashMap
from .flat_kmers import FlatKmers


class CollisionFreeKmerIndex:
    def __init__(self, hashes_to_index, n_kmers, nodes, ref_offsets, kmers, modulo=452930477):
        self._hashes_to_index = hashes_to_index
        self._n_kmers = n_kmers
        self._nodes = nodes
        self._ref_offsets = ref_offsets
        self._kmers = kmers  # Actual numeric kmers (not hashes of numeric kmers) at each position
                             # used to filter out collisions
        self._modulo = modulo

    def get(self, kmer):
        hash = kmer % self._modulo
        position = self._hashes_to_index[hash]
        n_hits = self._n_kmers[hash]
        start = position
        end = position + n_hits
        hit_positions = np.where(self._kmers[start:end] == kmer)[0]
        if len(hit_positions) == 0:
            return None, None
        return self._nodes[hit_positions + start], self._ref_offsets[hit_positions + start]

    def get_nodes_and_ref_offsets_from_multiple_kmers(self, kmers):
        all_nodes = []
        all_ref_offsets = []
        all_read_offsets = []
        for i, hash in enumerate(kmers):
            nodes, ref_offsets = self.get(hash)
            if nodes is None:
                continue
            all_nodes.append(nodes)
            all_ref_offsets.append(ref_offsets)
            all_read_offsets.append(np.zeros(len(nodes)) + i)

        if len(all_nodes) == 0:
            return np.array([]), np.array([]), np.array([])

        all_nodes = np.concatenate(all_nodes)
        all_ref_offsets = np.concatenate(all_ref_offsets)
        all_read_offsets = np.concatenate(all_read_offsets)
        return all_nodes, all_ref_offsets, all_read_offsets

    def to_file(self, file_name):
        logging.info("Writing kmer index to file: %s" % file_name)
        np.savez(file_name, hashes_to_index=self._hashes_to_index,
                 n_kmers=self._n_kmers,
                 nodes=self._nodes,
                 ref_offsets=self._ref_offsets,
                 kmers=self._kmers,
                 modulo=self._modulo)

    @classmethod
    def from_file(cls, file_name):
        data = np.load(file_name + ".npz")
        return cls(data["hashes_to_index"], data["n_kmers"], data["nodes"], data["ref_offsets"], data["kmers"], data["modulo"])

    @classmethod
    def from_flat_kmers(cls, flat_kmers, modulo=452930477):

        kmers = flat_kmers._hashes
        nodes = flat_kmers._nodes
        ref_offsets = flat_kmers._ref_offsets

        hashes = kmers % modulo
        sorting = np.argsort(hashes)
        hashes = hashes[sorting]
        kmers = kmers[sorting]
        nodes = nodes[sorting]
        ref_offsets = ref_offsets[sorting]

        # Find positions where hashes change (these are our index entries)
        diffs = np.ediff1d(hashes, to_begin=1)
        unique_entry_positions = np.nonzero(diffs)[0]
        unique_hashes = hashes[unique_entry_positions]

        lookup = np.zeros(modulo, dtype=np.int)
        lookup[unique_hashes] = unique_entry_positions
        n_entries = np.ediff1d(unique_entry_positions, to_end=len(nodes)-unique_entry_positions[-1])
        n_kmers = np.zeros(modulo, dtype=np.uint32)
        n_kmers[unique_hashes] = n_entries

        # Find out how many entries there are for each unique hash
        return cls(lookup, n_kmers, nodes, ref_offsets, kmers, modulo)




