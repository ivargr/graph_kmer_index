from graph_kmer_index import ReferenceKmerIndex
from graph_kmer_index import FlatKmers
import numpy as np

"""
def simple_test():
    ref = np.array([4, 4, 5, 5, 1, 2, 3], dtype=np.uint64)
    kmers = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.uint64)
    nodes = np.array([1, 2, 3, 4, 5, 6, 7])
    flat = FlatKmers(kmers, nodes, ref)

    index = ReferenceKmerIndex.from_flat_kmers(flat)
    index.to_file("testindex")
    index = index.from_file("testindex")

    result = index.get_between(1, 3)
    assert set(list(result)) == set([5, 6]), result
    assert list(index.get_between(1, 4)) == [5, 6, 7]
    assert list(index.get_between(2, 5)) == [6, 7, 1, 2]
    print(result)

    print(flat)



if __name__ == "__main__":
    simple_test()
"""