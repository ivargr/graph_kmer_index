from graph_kmer_index.nplist import NpList
import numpy as np


def test():
    list = NpList()
    list.append(5.0)
    list.append(10.0)

    assert np.all(list.get_nparray() == [5.0, 10.0])

    list2 = NpList(dtype=np.uint32)
    for i in range(10000):
        list2.append(i)

    array = list2.get_nparray()
    assert array.dtype == np.uint32
    assert len(array) == 10000
    assert len(list) == 2, len(list)
    assert len(list2) == 10000, len(list)


def test_extend():
    list = NpList()
    list.append(10.0)
    list.extend([1, 3, 4, 1, 5, 5])
    assert np.all(list.get_nparray() == [10.0, 1, 3, 4, 1, 5, 5])
    list.append(100)
    assert list[-1] == 100


def test_copy():
    l = NpList()
    l.append(10)
    l.append(100)
    l.extend(list(range(100)))

    l2 = l.copy()
    assert l2 == l
