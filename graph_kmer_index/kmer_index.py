import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from .logn_hash_map import LogNHashMap, ModuloHashMap
from .flat_kmers import FlatKmers


class KmerIndex:
    def __init__(self, hasher, hashes_to_index, n_kmers, nodes, ref_offsets):
        self._hasher = hasher
        self._hashes_to_index = np.array(hashes_to_index, dtype=np.int)
        self._n_kmers = n_kmers
        self._nodes = nodes
        self._ref_offsets = ref_offsets

    @classmethod
    def from_raw_arrays(cls, hashes, nodes, ref_offsets):
        pass

    def get(self, kmer_hash):
        index_hash = self._hasher.hash(kmer_hash)
        if index_hash == 0:
            return None
        if index_hash is None:
            return None

        position = self._hashes_to_index[index_hash]
        n_hits = self._n_kmers[index_hash]
        return self._nodes[position:position+n_hits]

    def get_nodes_and_ref_offsets(self, kmer_hash):
        index_hash = self._hasher.hash(kmer_hash)
        if index_hash == 0:
            return None, None
        if index_hash is None:
            return None, None

        position = self._hashes_to_index[index_hash]
        n_hits = self._n_kmers[index_hash]
        return self._nodes[position:position+n_hits], self._ref_offsets[position:position+n_hits]

    def get_nodes_and_ref_offsets_from_multiple_kmers(self, kmers):
        all_nodes = []
        all_ref_offsets = []
        all_read_offsets = []
        for i, hash in enumerate(kmers):
            nodes, ref_offsets = self.get_nodes_and_ref_offsets(hash)
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
        self._hasher.to_file(file_name + ".hasher")
        np.savez(file_name, hashes_to_index=self._hashes_to_index,
                 n_kmers=self._n_kmers,
                 nodes=self._nodes,
                 ref_offsets=self._ref_offsets)

    @classmethod
    def from_file(cls, file_name):
        hasher = ModuloHashMap.from_file(file_name + ".hasher")
        data = np.load(file_name + ".npz")
        return cls(hasher, data["hashes_to_index"], data["n_kmers"], data["nodes"], data["ref_offsets"])

    @classmethod
    def from_multiple_flat_kmers(cls, flat_kmers_list):

        hashes = []
        n_kmers = []
        nodes = []
        ref_offsets = []

        logging.info("Merging all")
        for flat_kmers in flat_kmers_list:
            hashes.append(flat_kmers._hashes)
            nodes.append(flat_kmers._nodes)
            ref_offsets.append(flat_kmers._ref_offsets)

        hashes = np.concatenate(hashes)
        nodes = np.concatenate(nodes)
        ref_offsets = np.concatenate(ref_offsets)

        sorted_indexes = np.argsort(hashes)
        hashes = hashes[sorted_indexes]
        #hasher = LogNHashMap(hashes)
        hasher = ModuloHashMap.from_sorted_array(hashes)
        nodes = nodes[sorted_indexes]
        ref_offsets = ref_offsets[sorted_indexes]

        # Get only positions of first new kmer in nodes. Hashes_to_index points to these positions
        diffs = np.ediff1d(hashes, to_begin=1)
        hashes_to_index = np.nonzero(diffs)[0]
        n_kmers = np.ediff1d(hashes_to_index, to_end=len(nodes)-hashes_to_index[-1])

        return cls(hasher, hashes_to_index, n_kmers, nodes, ref_offsets)


def test_from_single_flat_kmers():

    flat_kmers = FlatKmers([100, 150, 150, 300], [4, 5, 10, 6])
    index = KmerIndex.from_multiple_flat_kmers([flat_kmers])

    assert list(index.get(100)) == [4]
    assert list(index.get(150)) == [5, 10]
    assert list(index.get(300)) == [6]


def test_from_multiple_flat_kmers():
    flat_kmers = \
        [
            FlatKmers([654, 12, 554, 554], [5, 5, 4, 4]),
            FlatKmers([892, 123, 12, 432, 892, 123], [2, 3, 4, 6, 7, 100])
        ]
    index = KmerIndex.from_multiple_flat_kmers(flat_kmers)
    assert list(index.get(12)) == [5, 4]
    assert list(index.get(554)) == [4, 4]
    assert list(index.get(123)) == [3, 100]


if __name__ == "__main__":
    test_from_single_flat_kmers()
    test_from_multiple_flat_kmers()






