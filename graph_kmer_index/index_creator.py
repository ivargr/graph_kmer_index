import logging
from offsetbasedgraph import Interval, NumpyIndexedInterval
logging.basicConfig(level=logging.INFO)
import numpy as np


class IndexCreator:
    def __init__(self, graph, sequence_graph, vcfmap, k=15):
        self.graph = graph
        self.vcfmap = vcfmap
        self.sequence_graph = sequence_graph
        self.k = k


    def _make_haplotype_interval(self, haplotype):
        nodes = []
        first_nodes = self.graph.get_first_blocks()
        assert len(first_nodes) == 1
        current_node = first_nodes[0]
        i = 0
        while True:
            if i % 100000 == 0:
                logging.info("Traversed %d nodes. On node %d" % (i, current_node))
            i += 1
            if i > 1000000:
                break
            nodes.append(current_node)
            next_nodes = self.graph.adj_list[current_node]
            if len(next_nodes) == 0:
                break
            if len(next_nodes) == 1:
                current_node = next_nodes[0]
            else:
                for next in next_nodes:
                    haplotypes = self.vcfmap.get_haplotypes_on_edge(current_node, next)
                    if haplotypes is None:
                        current_node = next
                        break

                    if haplotype in haplotypes:
                        current_node = next
                        break

            assert current_node in next_nodes

        nodes.append(current_node)
        end_offset = self.graph.blocks[current_node].length()

        #return np.array(nodes)
        return Interval(0, end_offset, nodes, self.graph)

    def _get_numeric_interval_sequence(self, interval):
        #length = np.sum(self.graph.blocks._array[interval - ])
        sequences = []
        for node in interval:
            sequences.append(self.sequence_graph.get_numeric_node_sequence(node))

        sequences = np.concatenate(sequences)
        return sequences


    def _hash(self, numeric_sequence):
        n = 0
        for i, letter in enumerate(numeric_sequence):
            n += letter * 5**(self.k-i-i)
        return n

    def _dynamically_get_new_hash(self, previous_hash, previous_letter, next_letter):
        new_hash = previous_hash
        new_hash -= previous_letter * 5**(self.k-1)
        new_hash *= 5
        new_hash += next_letter
        return new_hash

    def _get_kmers(self, sequence, indexed_interval):
        kmers = np.zeros(len(sequence))
        current_kmer = self._hash(sequence[0:self.k])
        kmers[0] = current_kmer

        for i in range(0, len(sequence) - self.k):
            if i % 100000 == 0:
                logging.info("%d bases processed " % i)

            new_kmer = self._dynamically_get_new_hash(current_kmer, sequence[i], sequence[i+self.k])
            kmers[i] = new_kmer
            current_kmer = new_kmer

        return kmers

    def to_file(self, file_name):
        pass

    def create(self):
        #for haplotype in self.vcfmap.possible_haplotypes:
        for haplotype in [1]:

            logging.info("Running haplotype %d" % haplotype)
            interval = self._make_haplotype_interval(haplotype)
            logging.info("Done finding interval");
            #logging.info("Done with interval. Length: %d" % interval.length())
            indexed_interval = NumpyIndexedInterval.from_interval(interval)
            logging.info("Done indexing")
            sequence = self._get_numeric_interval_sequence(interval.region_paths)
            print(sequence)
            logging.info("Got sequence")
            logging.info("Sequence length: %d" % len(sequence))
            self._get_kmers(sequence, indexed_interval)

