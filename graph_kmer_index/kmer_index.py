import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from .logn_hash_map import LogNHashMap
from .flat_kmers import FlatKmers


class KmerIndex:
    def __init__(self, hasher, hashes_to_index, n_kmers, nodes):
        self._hasher = hasher
        self._hashes_to_index = hashes_to_index
        self._n_kmers = n_kmers
        self._nodes = nodes

    @classmethod
    def from_raw_arrays(cls, hashes, nodes):
        pass

    def get(self, kmer_hash):
        index_hash = self._hasher.hash(kmer_hash)
        print("Index hash: ", index_hash)
        if index_hash is None:
            return None

        position = self._hashes_to_index[index_hash]
        n_hits = self._n_kmers[index_hash]
        return self._nodes[position:position+n_hits]

    def to_file(self, file_name):
        logging.info("Writing kmer index to file: %s" % file_name)
        self._hasher.to_file(file_name + ".hasher")
        np.savez(file_name, hashes_to_index=self._hashes_to_index,
                 n_kmers=self._n_kmers,
                 nodes=self._nodes)

    @classmethod
    def from_file(cls, file_name):
        hasher = LogNHashMap.from_file(file_name + ".hasher")
        data = np.load(file_name + ".npz")
        return cls(hasher, data["hashes_to_index"], data["n_kmers"], data["nodes"])

    @classmethod
    def from_multiple_flat_kmers(cls, flat_kmers_list):

        hashes = []
        n_kmers = []
        nodes = []

        logging.info("Merging all")
        for flat_kmers in flat_kmers_list:
            hashes.append(flat_kmers._hashes)
            nodes.append(flat_kmers._nodes)

        hashes = np.concatenate(hashes)
        nodes = np.concatenate(nodes)

        sorted_indexes = np.argsort(hashes)
        hashes = hashes[sorted_indexes]
        hasher = LogNHashMap(hashes)
        nodes = nodes[sorted_indexes]

        # Get only positions of first new kmer in nodes. Hashes_to_index points to these positions
        diffs = np.ediff1d(hashes, to_begin=1)
        hashes_to_index = np.nonzero(diffs)[0]
        n_kmers = np.ediff1d(hashes_to_index, to_end=len(nodes)-hashes_to_index[-1])

        return cls(hasher, hashes_to_index, n_kmers, nodes)


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






