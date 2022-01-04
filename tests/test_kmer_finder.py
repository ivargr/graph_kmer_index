import logging
from graph_kmer_index.kmer_finder import DenseKmerFinder
from obgraph import Graph
logging.basicConfig(level=logging.INFO)



def simple_test():

    graph = Graph.from_dicts(
        {0: "ACTGACTG", 1: "A", 2: "T", 3: "AAAAA", 4: "C", 5: "T", 6: "GGGGGG"},
        {0: [1, 2], 2: [3], 1: [3], 3: [4, 5], 4: [6], 5: [6]},
        [0, 1, 3, 4, 6]
    )
    graph.set_numeric_node_sequences()
    print(graph.numeric_node_sequences)
    finder = DenseKmerFinder(graph, k=3)
    finder.find()

    print(finder.results)


simple_test()