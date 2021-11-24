import numpy as np
import logging
import pickle

from graph_kmer_index import kmer_hash_to_sequence, sequence_to_kmer_hash
from Bio.Seq import Seq

class CollisionFreeKmerIndex:
    properties = {
            "_hashes_to_index",
            "_n_kmers",
            "_nodes",
            "_ref_offsets",
            "_kmers",
            "_modulo",
            "_frequencies",
            "_allele_frequencies"
        }

    def __init__(self, hashes_to_index=None, n_kmers=None, nodes=None, ref_offsets=None, kmers=None, modulo=452930477, frequencies=None, allele_frequencies=None):
        self._hashes_to_index = hashes_to_index
        self._n_kmers = n_kmers
        self._nodes = nodes
        self._ref_offsets = ref_offsets
        self._kmers = kmers  # Actual numeric kmers (not hashes of numeric kmers) at each position
                             # used to filter out collisions
        self._modulo = modulo
        if frequencies is None:
            self._frequencies = 0
        else:
            self._frequencies = frequencies

        self._allele_frequencies = allele_frequencies

    def set_allele_frequencies(self, frequencies):
        pass

    def max_node_id(self):
        return np.max(self._nodes)

    def set_frequencies_using_other_index(self, other, multiplier=1, min_frequency=1):
        unique = np.unique(self._kmers)
        for i, kmer in enumerate(unique):
            kmer = int(kmer)
            if i % 100000 == 0:
                logging.info("%d/%d unique kmers processed" % (i, len(unique)))
            frequency = other.get_frequency(kmer)
            hash = int(kmer) % self._modulo
            position = self._hashes_to_index[hash]
            n_hits = self._n_kmers[hash]
            start = position
            end = position + n_hits
            hit_positions = np.where(self._kmers[start:end] == kmer)[0]
            self._frequencies[hit_positions + start] = max(min_frequency, frequency * multiplier)

    def set_frequencies(self, skip=False):
        logging.info("Setting frequencies")
        # Count number of each kmer (not hashes, store these)
        self._frequencies = np.zeros(len(self._kmers), dtype=np.uint16)

        if skip:
            logging.info("Skipped setting frequencies. All frequencies are just 0 by default.")
            return

        unique = np.unique(self._kmers)
        for i, kmer in enumerate(unique):
            if i % 100000 == 0:
                logging.info("%d/%d unique kmers processed" % (i, len(unique)))

            hash = int(kmer) % self._modulo
            position = self._hashes_to_index[hash]
            n_hits = self._n_kmers[hash]
            start = position
            #assert start != 0 or hash == 0, "Kmer %d with hash %d, index position %d not found in index" % (kmer, hash, position)
            end = position + n_hits
            hit_positions = np.where(self._kmers[start:end] == kmer)[0]

            # The count is the number of unique ref positions here
            # (since same entry can have multiple nodes, but always same ref pos)
            count = len(set(self._ref_offsets[hit_positions + start]))
            assert count > 0, "Count is not > 0 for kmer %d, start, end: %d,%d. Ref offsets: %s" % (kmer, start, end, self._ref_offsets[hit_positions + start])
            self._frequencies[hit_positions + start] = count

    def get(self, kmer, max_hits=10):
        hash = kmer % self._modulo
        position = self._hashes_to_index[hash]
        n_hits = self._n_kmers[hash]
        start = position
        end = position + n_hits
        hit_positions = np.where(self._kmers[start:end] == kmer)[0]
        frequencies = self._frequencies[hit_positions+start]
        allele_frequencies = self._allele_frequencies[hit_positions+start]
        if len(hit_positions) == 0 or frequencies[0] > max_hits:
            return None, None, None, None

        return self._nodes[hit_positions + start], self._ref_offsets[hit_positions + start], frequencies, allele_frequencies

    def get_grouped_nodes(self, kmer, max_hits=10):
        hits = self.get(kmer, max_hits)
        if hits[0] is None:
            return None

        ref_offsets = hits[1]
        nodes = hits[0]
        sorting = np.argsort(ref_offsets)
        ref_offsets = ref_offsets[sorting]
        nodes = nodes[sorting]


        _, hit_indexes = np.unique(ref_offsets, return_index=True)
        hit_indexes = list(hit_indexes)
        hit_indexes.append(len(ref_offsets))

        intervals = [(start, end) for start, end in zip(hit_indexes[0:-1], hit_indexes[1:])]
        return [nodes[start:end] for start, end in intervals]

    def get_frequency(self, kmer, include_reverse_complement=True, k=31):
        nodes, ref_offsets, frequencies, allele_frequencies = self.get(kmer, max_hits=1000000000000000)
        if nodes is None:
            f = 0
        else:
            f = int(frequencies[0])  # convert to avoid overflow error

        if include_reverse_complement:
            sequence = kmer_hash_to_sequence(kmer, k)
            rev_sequence = str(Seq(sequence).reverse_complement())
            rev_kmer = sequence_to_kmer_hash(rev_sequence)
            nodes, ref_offsets, frequencies, allele_frequencies = self.get(rev_kmer, max_hits=1000000000000000)

            if nodes is not None:
                f += int(frequencies[0])

        return f

    def get_nodes_and_ref_offsets_from_multiple_kmers(self, kmers, max_hits=10):
        all_nodes = []
        all_ref_offsets = []
        all_read_offsets = []
        all_frequencies = []
        for i, hash in enumerate(kmers):
            nodes, ref_offsets, frequencies, allele_frequencies = self.get(hash, max_hits=max_hits)
            if nodes is None:
                continue
            all_nodes.append(nodes)
            all_ref_offsets.append(ref_offsets)
            all_read_offsets.append(np.zeros(len(nodes)) + i)
            all_frequencies.append(frequencies)


        if len(all_nodes) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        all_nodes = np.concatenate(all_nodes)
        all_ref_offsets = np.concatenate(all_ref_offsets)
        all_read_offsets = np.concatenate(all_read_offsets)
        all_frequencies = np.concatenate(all_frequencies)
        return all_nodes, all_ref_offsets, all_read_offsets, all_frequencies

    def get_nodes_from_multiple_kmers(self, kmers, max_hits=10):
        all_nodes = []
        for i, hash in enumerate(kmers):
            nodes, ref_offsets, frequencies, allele_frequencies = self.get(hash, max_hits=max_hits)
            if nodes is None:
                continue
            all_nodes.append(nodes)


        if len(all_nodes) == 0:
            return np.array([])

        all_nodes = np.concatenate(all_nodes)
        return all_nodes

    def to_file(self, file_name):
        logging.info("Writing kmer index to file: %s" % file_name)
        np.savez(file_name, hashes_to_index=self._hashes_to_index,
                 n_kmers=self._n_kmers,
                 nodes=self._nodes,
                 ref_offsets=self._ref_offsets,
                 kmers=self._kmers,
                 modulo=self._modulo,
                 frequencies=self._frequencies,
                 allele_frequencies=self._allele_frequencies)

    @classmethod
    def from_file(cls, file_name):
        try:
            data = np.load(file_name + ".npz")
        except FileNotFoundError:
            data = np.load(file_name)

        if "allele_frequencies" in data:
            allele_frequencies = data["allele_frequencies"]
        else:
            allele_frequencies = np.zeros(len(data["ref_offsets"]))

        return cls(data["hashes_to_index"], data["n_kmers"], data["nodes"], data["ref_offsets"], data["kmers"], data["modulo"], data["frequencies"], allele_frequencies)

    @classmethod
    def from_flat_kmers(cls, flat_kmers, modulo=452930477, skip_frequencies=False, skip_singletons=False):
        if skip_singletons:
            flat_kmers = flat_kmers.get_new_without_singletons()

        kmers = flat_kmers._hashes
        nodes = flat_kmers._nodes
        ref_offsets = flat_kmers._ref_offsets

        logging.info("Making hashes")
        hashes = kmers % modulo
        logging.info("Sorting")
        sorting = np.argsort(hashes)
        hashes = hashes[sorting]
        kmers = kmers[sorting]
        nodes = nodes[sorting]
        ref_offsets = ref_offsets[sorting]
        allele_frequencies = flat_kmers._allele_frequencies[sorting]
        logging.info("Done sorting")

        # Find positions where hashes change (these are our index entries)
        diffs = np.ediff1d(hashes, to_begin=1)
        unique_entry_positions = np.nonzero(diffs)[0]
        try:
            unique_hashes = hashes[unique_entry_positions]
        except IndexError:
            logging.info("unique entry positions: %s" % unique_entry_positions)
            logging.info("Hashes: %s" % hashes)
            raise

        lookup = np.zeros(modulo, dtype=np.int)
        lookup[unique_hashes] = unique_entry_positions
        n_entries = np.ediff1d(unique_entry_positions, to_end=len(nodes)-unique_entry_positions[-1])
        n_kmers = np.zeros(modulo, dtype=np.uint32)
        n_kmers[unique_hashes] = n_entries

        # Find out how many entries there are for each unique hash
        object = cls(lookup, n_kmers, nodes, ref_offsets, kmers, modulo, allele_frequencies=allele_frequencies)
        object.set_frequencies(skip_frequencies)

        if skip_singletons:
            logging.info("Adding 1 to all frequencies since singletons are skipped")
            object._frequencies += 1

        return object





