from graph_kmer_index.cython_kmer_index import CythonKmerIndex
from graph_kmer_index import CollisionFreeKmerIndex
import numpy as np


def test_same_result_as_non_cython():
    kmers =  [3608911655527732, 3457176383516950184]
    for kmer in kmers:
        print("Checking kmer %s " % kmer)
        i = CollisionFreeKmerIndex.from_file("kmer_index31.npz")
        c = CythonKmerIndex(i)
        res = c.get(np.array([kmer], dtype=np.uint64))
        print("REs: ", res)
        node_hits = list(res[0, :])
        ref_offsets = list(res[1, :])
        read_offsets = list(res[2, :])
        print(node_hits)
        print(ref_offsets)

        assert node_hits == list(i.get(kmer)[0])
        assert ref_offsets == list(i.get(kmer)[1])
        assert read_offsets == [0, 0]


if __name__ == "__main__":
    test_same_result_as_non_cython()
