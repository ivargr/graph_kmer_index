import itertools
import logging
import numpy as np
from graph_kmer_index import kmer_to_hash_fast, letter_sequence_to_numeric
from Bio.Seq import Seq
from .kmer_hashing import power_array


class ReadKmers:
    def __init__(self, kmers):
        self.kmers = kmers
        self._power_vector = None

    @classmethod
    def from_fasta_file(cls, fasta_file_name, k, small_k=None, smallest_k=8):
        power_vector = power_array(k)
        f = open(fasta_file_name)
        f = [l for l in f.readlines() if not l.startswith(">")]
        logging.info("Number of lines: %d" % len(f))
        if small_k is None:
            kmers = itertools.chain(
                (ReadKmers.get_kmers_from_read_dynamic(line.strip(), power_vector)
                        for line in f if not line.startswith(">")),
                (ReadKmers.get_kmers_from_read_dynamic(str(Seq(line.strip()).reverse_complement()), power_vector)
                        for line in f if not line.startswith(">"))
            )
        else:
            power_vector_small = power_array(small_k)
            power_vector_smallest = power_array(smallest_k)
            kmers = zip(
                    (itertools.chain(

                        ReadKmers.get_kmers_from_read_dynamic(line.strip(), power_vector),
                        ReadKmers.get_kmers_from_read_dynamic(str(Seq(line.strip()).reverse_complement()), power_vector)

                    ) for line in f),
                    (itertools.chain(
                            ReadKmers.get_kmers_from_read_dynamic(line.strip(), power_vector_small),
                            ReadKmers.get_kmers_from_read_dynamic(str(Seq(line.strip()).reverse_complement()), power_vector_small)
                    )
                    for line in f),
                    (itertools.chain(
                        ReadKmers.get_kmers_from_read_dynamic(line.strip(), power_vector_smallest),
                        ReadKmers.get_kmers_from_read_dynamic(str(Seq(line.strip()).reverse_complement()), power_vector_smallest)
                    )
                    for line in f)
            )

        return cls(kmers)

    @classmethod
    def from_list_of_string_kmers(cls, string_kmers):
        kmers = [
            [kmer_to_hash_fast(letter_sequence_to_numeric(k), len(k)) for k in read_kmers]
            for read_kmers in string_kmers
        ]
        return cls(kmers)

    @staticmethod
    def get_kmers_from_read(read, k):
        kmers = []
        for i in range(len(read) - k):
            letter_sequence = letter_sequence_to_numeric(read[i:i+k])
            kmers.append(kmer_to_hash_fast(letter_sequence, k))
        return kmers

    @staticmethod
    def get_kmers_from_read_dynamic(read, power_vector):
        read = letter_sequence_to_numeric(read)
        return np.convolve(read, power_vector, mode='valid')

    @staticmethod
    def get_kmers_from_read_dynamic_slow(read, k):
        raise NotImplementedError()
        read = letter_sequence_to_numeric(read)
        kmers = np.zeros(len(read)-k+1, dtype=np.int64)
        current_hash = kmer_to_hash_fast(read[0:k], k)
        kmers[0] = current_hash
        for i in range(1, len(read)-k+1):
            kmers[i] = (kmers[i-1] - np.power(4, k-1) * read[i-1]) * 4 + read[i+k-1]
            #assert kmers[i] == kmer_to_hash_fast(read[i:i+k], k), "New hash %d != correct %d" % (kmers[i], kmer_to_hash_fast(read[i:i+k], k))

        return kmers

    def __iter__(self):
        return self.kmers.__iter__()

    def __next__(self):
        return self.kmers.__next__()