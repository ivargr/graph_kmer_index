import pickle
import logging
import numpy as np
import itertools
from collections import defaultdict


class UniqueKmerIndex:
    def __init__(self, index_dict):
        self._index_dict = index_dict

    def to_file(self, file_name):
        file = open(file_name, "wb")
        pickle.dump(self._index_dict, file)

    @classmethod
    def from_file(cls, file_name):
        with open(file_name, "rb") as f:
            data = pickle.load(f)

        return cls(data)

    def get(self, kmer):
        return self._index_dict[kmer]

    @classmethod
    def from_flat_kmers_and_snps_graph(cls, flat_kmers, graph, reverse_index):
        g = graph
        flat = flat_kmers

        logging.info("Finding kmers in flat index that are unique in regards to starting position and kmers")
        no_duplicates = []
        found = set()

        for i in range(len(flat._hashes)):
            hash = flat._hashes[i]
            ref_pos = flat._ref_offsets[i]

            if (hash, ref_pos) not in found:
                no_duplicates.append(hash)
                found.add((hash, ref_pos))

            if i % 100000 == 0:
                print("%d of %d kmers procsesed" % (i, len(flat._hashes)))

        logging.info("Number of no kmers that are not duplicate in index: %d" % len(no_duplicates))
        no_duplicates = np.array(no_duplicates)

        logging.info("Finding snps in graph")
        snps = [g.adj_list[node] for node in g.blocks if len(g.adj_list[node]) == 2]
        logging.info("Finding kmers that occur only once in index")
        unique_no_duplicates, positions = np.unique(no_duplicates, return_counts=True)
        unique_kmers = set(unique_no_duplicates[np.where(positions == 1)[0]])

        logging.info("Creating index as SNPs where both nodes only have kmers thare are unique in index")

        has_unique = [np.all([kmer in unique_kmers for kmer in
                              itertools.chain(reverse_index.get_node_kmers(nodes[0]), reverse_index.get_node_kmers(nodes[1]))])
                      if len(reverse_index.get_node_kmers(nodes[0])) < 4 and len(reverse_index.get_node_kmers(nodes[1])) < 4 else False
                      for
                      nodes in snps]

        snps_array = np.array(snps)
        snps_with_unique_kmers = snps_array[np.array(has_unique)]

        unique_index = defaultdict(list)
        for i, node_pair in enumerate(snps_with_unique_kmers):
            for node in node_pair:
                for kmer in reverse_index.get_node_kmers(node):
                    unique_index[kmer].append(node)

            if i % 1000 == 0:
                logging.info("Creating index. %d/%d variants processed " % (len(unique_index), len(snps_with_unique_kmers)))


        return cls(unique_index)
