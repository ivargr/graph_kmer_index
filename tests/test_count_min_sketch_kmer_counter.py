import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from graph_kmer_index.count_min_sketch_kmer_counter import CountMinSketchKmerCounter


def test_simple():
    counter = CountMinSketchKmerCounter.create_empty([3, 9, 13])
    counter.count_kmers(np.array([123, 5]))
    assert counter.get_count(123) == 1
    assert counter.get_count(5) == 1
    counter.count_kmers(np.array([5, 5, 5]))
    assert counter.get_count(5) == 4


test_simple()