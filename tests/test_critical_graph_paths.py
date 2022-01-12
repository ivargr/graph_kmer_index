from graph_kmer_index.critical_graph_paths import CriticalGraphPaths
from obgraph import Graph
import numpy as np


def test():
    graph = Graph.from_dicts(
        {0: "AAA", 1: "C", 2: "T", 3: "AAA"},
        {0: [1, 2], 2: [3], 1: [3]},
        [0, 1, 3]
    )

    k = 3

    c = CriticalGraphPaths.from_graph(graph, k)

    assert np.all(c.nodes == [0, 3])
    assert np.all(c.offsets == [2, 2])

    k = 4
    c = CriticalGraphPaths.from_graph(graph, k)
    assert len(c.nodes) == 0



def test2():
    graph = Graph.from_dicts(
        {0: "AAACCCTTTT", 1: "CTTT", 2: "TAAGGGG", 3: "AAA"},
        {0: [1, 2], 2: [3], 1: [3]},
        [0, 1, 3]
    )

    k = 3

    c = CriticalGraphPaths.from_graph(graph, k)
    assert np.all(c.nodes == [0, 3])
    assert np.all(c.offsets == [2, 2])


def test3():
    graph = Graph.from_dicts(
        {0: "ACTGACTG", 1: "A", 2: "T", 3: "AAAAA", 4: "C", 5: "T", 6: "TGGGGG"},
        {0: [1, 2], 2: [3], 1: [3], 3: [4, 5], 4: [6], 5: [6]},
        [0, 1, 3, 4, 6]
    )

    k = 3
    c = CriticalGraphPaths.from_graph(graph, k)
    assert np.all(c.nodes == [0, 3, 6])
    assert np.all(c.offsets == [2, 2, 2])


def test4():
    graph = Graph.from_dicts(
        {0: "A", 1: "CTTT", 2: "TAAGGGG", 3: "AAA"},
        {0: [1], 1: [2], 2: [3], 1: [3]},
        [0, 1, 2, 3]
    )

    k = 3
    c = CriticalGraphPaths.from_graph(graph, k)
    assert np.all(c.nodes == [1])
    assert np.all(c.offsets == [1])


def test5():
    graph = Graph.from_dicts(
        {0: "ACTGACTG", 1: "A", 2: "T", 3: "AAAAA", 4: "C", 5: "T", 6: "TGGGGG", 100: ""},
        {0: [1, 2, 100], 2: [3], 1: [3], 3: [4, 5], 4: [6], 5: [6], 100: [6]},
        [0, 1, 3, 4, 6]
    )
    graph.make_linear_ref_node_and_ref_dummy_node_index()

    k = 3
    c = CriticalGraphPaths.from_graph(graph, k)
    assert np.all(c.nodes == [0, 6])
    assert np.all(c.offsets == [2, 2])



test()
test2()
test3()
test4()
test5()