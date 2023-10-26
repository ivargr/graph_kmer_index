from graph_kmer_index.flat_kmers import FlatKmers
from graph_kmer_index.collision_free_kmer_index import CollisionFreeKmerIndex
import numpy as np
import pytest

@pytest.fixture
def index():
    flat = FlatKmers(
        np.array([1, 1, 2, 2, 4, 5, 3], dtype=np.uint64),
        np.array([5, 6, 7, 8, 10, 11, 100]),
        np.array([1, 1, 2, 3, 10, 11, 100])
    )

    index = CollisionFreeKmerIndex.from_flat_kmers(flat, modulo=4)
    return index

def test_simple(index):

    assert list(index.get(1)[0]) == [5, 6]
    assert list(index.get(1)[1]) == [1, 1]

    index.to_file("tmp.index")
    index = CollisionFreeKmerIndex.from_file("tmp.index")

    assert list(index.get(5)[0]) == [11]
    print(index.get(3))
    print(index.get_nodes_and_ref_offsets_from_multiple_kmers(np.array([1, 5])))


def test_has_kmers_parallel(index):
    index.convert_to_int32()
    kmers = np.array([1, 2, 3, 10, 10, 12, 100, 101, 102, 5], dtype=np.uint64)
    result = index.has_kmers_parallel(kmers, n_threads=3)
    assert np.all(result == [True, True, True, False, False, False, False, False, False, True]), result
