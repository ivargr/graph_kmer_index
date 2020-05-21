from graph_kmer_index.cython_kmer_index import CythonKmerIndex
from graph_kmer_index import CollisionFreeKmerIndex
import numpy as np


def test_same_result_as_non_cython():
    kmer = 3608911655527732
    i = CollisionFreeKmerIndex.from_file("kmer_index31.npz")
    c = CythonKmerIndex(i)
    c.get(np.array([kmer], dtype=np.uint64))
    node_hits = list(c.get_node_hits())
    ref_offsets = list(c.get_ref_offsets_hits())
    read_offsets = list(c.get_read_offsets_hits())
    print(node_hits)
    print(ref_offsets)

    assert node_hits == list(i.get(kmer)[0])
    assert ref_offsets == list(i.get(kmer)[1])
    assert read_offsets == [0, 0]


if __name__ == "__main__":
    test_same_result_as_non_cython()
