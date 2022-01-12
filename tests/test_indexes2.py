from graph_kmer_index.collision_free_kmer_index import KmerIndex2
from graph_kmer_index.flat_kmers import FlatKmers2
import numpy as np


def test():
    flat_kmers = FlatKmers2(
        np.array([1, 1, 1, 2, 3, 10, 11, 2]),
        np.array([1, 1, 2, 2, 3, 1, 10,  5]), # start nodes
        np.array([0, 0, 1, 2, 3, 4, 5, 6]), # start offsets
        np.array([1, 2, 3, 4, 5, 6, 7, 8]), # nodes
        np.array([0.4, 0.1, 0.3, 0.4, 0.1, 0.1, 0.1, 0.1])
    )

    index = KmerIndex2.from_flat_kmers(flat_kmers)

    assert index.get_kmer_frequency(1) == 2
    assert np.all(index.get_start_nodes(1) == [1, 1, 2])
    assert np.all(index.get_nodes(3) == [5])


test()