from graph_kmer_index import FlatKmers, ReverseKmerIndex
import numpy as np


def test_from_flat_kmers():

    flat = FlatKmers(np.array([10, 3, 11, 4]), np.array([5, 3, 5, 8]))
    reverse = ReverseKmerIndex.from_flat_kmers(flat)
    print(reverse)

    assert 11 in reverse.get_node_kmers(5)
    assert 10 in reverse.get_node_kmers(5)
    assert 3 in reverse.get_node_kmers(3)
    assert 4 in reverse.get_node_kmers(8)

    reverse.to_file("tmp.reverse")
    new_reverse = ReverseKmerIndex.from_file("tmp.reverse.npz")

    assert 3 in new_reverse.get_node_kmers(3)

if __name__ == "__main__":
    test_from_flat_kmers()
