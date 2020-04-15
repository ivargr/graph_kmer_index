from graph_kmer_index.flat_kmers import FlatKmers
from graph_kmer_index.collision_free_kmer_index import CollisionFreeKmerIndex
import numpy as np


def test_simple():
    flat = FlatKmers(
        np.array([1, 1, 2, 2, 4, 5, 3]),
        np.array([5, 6, 7, 8, 10, 11, 100]),
        np.array([1, 1, 2, 3, 10, 11, 100])
    )

    index = CollisionFreeKmerIndex.from_flat_kmers(flat, modulo=4)
    assert list(index.get(1)[0]) == [5, 6]
    assert list(index.get(1)[1]) == [1, 1]

    index.to_file("tmp.index")
    index = CollisionFreeKmerIndex.from_file("tmp.index")

    assert list(index.get(5)[0]) == [11]
    print(index.get(3))
    print(index.get_nodes_and_ref_offsets_from_multiple_kmers(np.array([1, 5])))


if __name__ == "__main__":
    test_simple()
