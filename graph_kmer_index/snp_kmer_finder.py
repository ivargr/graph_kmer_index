import logging
import time
from .kmer_hashing import power_array

logging.basicConfig(level=logging.INFO)
import numpy as np
from .flat_kmers import FlatKmers, letter_sequence_to_numeric
from Bio.Seq import Seq
from collections import defaultdict
from .kmer_hashing import reverse_power_array, power_array, kmer_hashes_to_bases
from .flat_kmers import numeric_to_letter_sequence


def kmer_hash_to_sequence(hash, k):
    bases = kmer_hashes_to_bases(np.array([hash]), k)[0]
    return ''.join([b for b in numeric_to_letter_sequence(bases)])


def sequence_to_kmer_hash(sequence):
    return kmer_to_hash_fast(letter_sequence_to_numeric(sequence).astype(np.uint64), len(sequence))


#@jit(nopython=True)
def kmer_to_hash_fast(kmer, k):
    assert kmer.dtype == np.uint64
    return int(np.sum(kmer * reverse_power_array(k)))


class SnpKmerFinder:
    """
    Simple kmer finder that only supports SNP graphs
    """

    def __init__(self, graph, k=15, spacing=None, include_reverse_complements=False, pruning=False, max_kmers_same_position=100000,
                 max_frequency=10000, max_variant_nodes=10000, only_add_variant_kmers=False, whitelist=None, only_save_variant_nodes=False,
                 start_position=None, end_position=None, only_store_nodes=None, skip_kmers_with_nodes=None, only_save_one_node_per_kmer=False,
                 reference=None, variant_to_nodes=None, node_to_variants=None, haplotype_matrix=None):
        self.graph = graph
        self.reference = reference
        self.k = k
        #logging.info("Getting linear ref nodes")
        #self.linear_nodes = graph.linear_ref_nodes()


        self._hashes = []
        self._nodes = []
        self._ref_offsets = []
        self._allele_frequencies = []
        self.kmers_found = []
        self._bases_in_search_path = []
        self._nodes_in_path = []
        self._kmers_found = 0
        self._current_ref_offset = None
        self._last_ref_pos_added = 0
        self.pruning = pruning
        self._n_kmers_skipped_low_allele_frequency = 0
        self._n_kmers_pruned = 0
        self._has_traversed_variant = False
        self._unique_kmers_added = set()
        self._max_kmers_same_position = max_kmers_same_position
        self._n_kmers_added_current_position = 0
        self._n_kmers_skipped = 0
        self._kmer_frequencies = defaultdict(int)
        self._max_frequency = max_frequency
        self._n_skipped_due_to_frequency = 0
        self._max_variant_nodes = max_variant_nodes
        self._n_skipped_due_to_max_variant_nodes = 0
        self._only_add_variant_kmers = only_add_variant_kmers
        self._whitelist = whitelist
        self._n_skipped_whitelist = 0
        self._start_position = start_position
        self._end_position = end_position
        self._only_store_nodes = only_store_nodes
        self._skip_kmers_with_nodes = skip_kmers_with_nodes
        self._n_skipped_blacklist_nodes = 0
        self._only_save_one_node_per_kmer = only_save_one_node_per_kmer
        self.haplotype_matrix = haplotype_matrix
        self.variant_to_nodes = variant_to_nodes
        self.node_to_variants = node_to_variants

        if self._start_position == None:
            self._start_position = 0

        self._only_save_variant_nodes = only_save_variant_nodes
        self._variant_nodes = set()
        if self._only_save_variant_nodes:
            logging.info("Will only store nodes associated to variants for each kmer")
            # Find all variant nodes, store in a set
            for node in range(len(graph.nodes)):
                if node % 100000 == 0:
                    logging.info("Adding variant nodes, on node %d" % node)
                if len(graph.get_edges(node)) > 1:
                    for next_node in graph.get_edges(node):
                        self._variant_nodes.add(next_node)

        if spacing is None:
            self.spacing = k
        else:
            self.spacing = spacing
        self._include_reverse_complements=include_reverse_complements
        if self._include_reverse_complements:
            logging.info("Will include reverse complement of kmers")


    def has_kmer(self, kmer, nodes):
        for found in self.kmers_found:
            if found[0] == kmer and found[1] == nodes:
                return True
        return False


        if (kmer, nodes) in self.kmers_found:
            return True
        return False


    def _add_kmer(self, kmer, nodes):
        #logging.info("      Addign kmer %s with nodes %s" % (kmer, nodes))
        self._n_paths_searched += 1

        assert len(kmer) == self.k

        hash = kmer_to_hash_fast(letter_sequence_to_numeric(kmer), k=len(kmer))

        if self._whitelist is not None:
            # Either hash or reverse complement hash can be in whitelist in order to keep this kmer
            if hash not in self._whitelist:
                rev_hash = kmer_to_hash_fast(letter_sequence_to_numeric(str(Seq(kmer).reverse_complement())),
                                             k=len(kmer))
                if rev_hash not in self._whitelist:
                    self._n_skipped_whitelist += 1
                    return

        if self._skip_kmers_with_nodes is not None and len(set(nodes).intersection(self._skip_kmers_with_nodes)) > 0:
            self._n_skipped_blacklist_nodes += 1
            return

        if not self._has_traversed_variant and self._only_add_variant_kmers:
            return

        if self._kmer_frequencies[hash] >= self._max_frequency:
            self._n_skipped_due_to_frequency += 1
            return

        if self.pruning and hash not in self._unique_kmers_added:
            if self._last_ref_pos_added != self._current_ref_offset and self._last_ref_pos_added > self._current_ref_offset - 124:
                if not self._has_traversed_variant:
                    # Do not add
                    self._n_kmers_pruned += 1
                    return

        if self._n_kmers_added_current_position > self._max_kmers_same_position:
            self._n_kmers_skipped += 1
            return

        n_variant_nodes = len([n for n in nodes if not self.graph.is_linear_ref_node_or_linear_ref_dummy_node(n)])
        if n_variant_nodes >= self._max_variant_nodes:
            self._n_skipped_due_to_max_variant_nodes += 1
            logging.info("SKIPPING KMER DUE TO TOO MANY VARIANT NODES")
            return

        if self._include_reverse_complements:
            rev_hash = kmer_to_hash_fast(letter_sequence_to_numeric(str(Seq(kmer).reverse_complement())), k=len(kmer))

        self._unique_kmers_added.add(hash)
        self._kmer_frequencies[hash] += 1


        if self.haplotype_matrix is not None:
            kmer_allele_frequency = self.haplotype_matrix.get_allele_frequency_for_nodes(nodes, self.node_to_variants, self.variant_to_nodes)
        else:
            kmer_allele_frequency = min([self.graph.get_node_allele_frequency(node) for node in nodes])

        for node in nodes:
            if self._only_save_variant_nodes and node not in self._variant_nodes:
                continue

            if self._only_store_nodes is not None and node not in self._only_store_nodes:
                continue

            self._hashes.append(hash)
            self._nodes.append(node)
            self._ref_offsets.append(self._current_ref_offset)
            self._allele_frequencies.append(kmer_allele_frequency)

            if self._include_reverse_complements:
                self._hashes.append(rev_hash)
                self._nodes.append(node)
                self._ref_offsets.append(self._current_ref_offset)
                self._allele_frequencies.append(kmer_allele_frequency)

            if self._only_save_one_node_per_kmer:
                break

        self._last_ref_pos_added = self._current_ref_offset

        self._kmers_found += 1
        if len(self.kmers_found) < 500:
            # Only add to this when there is little data, only used for testing and debugging
            #loggign.info("------Added kmer with nodes %s" % nodes)
            self.kmers_found.append((kmer, nodes, self._current_ref_offset, hash))

        self._n_kmers_added_current_position += 1

    def _find_all_variant_kmers_from_position(self, linear_ref_pos):
        self._n_paths_searched = 0
        self._n_variant_nodes_passed = 0
        self._current_ref_offset = linear_ref_pos

        # Always start one base pair before, but do not include that base pair
        # this lets us catch cases where we are at a beginning of a node

        if linear_ref_pos > 0:
            node = self.graph.get_node_at_ref_offset(linear_ref_pos-1)
            offset = self.graph.get_node_offset_at_ref_offset(linear_ref_pos-1) + 1
        else:
            node = self.graph.get_node_at_ref_offset(linear_ref_pos)
            offset = self.graph.get_node_offset_at_ref_offset(linear_ref_pos)

        self._bases_in_search_path = []
        self._nodes_in_path = []
        self._has_traversed_variant = False
        self._n_kmers_added_current_position = 0
        self._search_graph_from(node, offset, self.k)



    def _search_graph_from(self, node, offset, bases_left):
        #loggign.info("== Searching node %d from offset %d. Bases left: %d == " % (node, offset, bases_left))

        if False and self._n_paths_searched > 1000:
            logging.warning("More than 1000 paths searched from ref pos %d. Possibly many variants here?" % self._current_ref_offset)

        if bases_left == 0:
            self._add_kmer(''.join(self._bases_in_search_path).replace("-", ""), set(self._nodes_in_path))
            #loggign.info("Recursion is done, added kmer with nodes: %s" % str(self._nodes_in_path))
            return

        if not self.graph.is_linear_ref_node_or_linear_ref_dummy_node(node):
            self._n_variant_nodes_passed += 1

        # Process the rest of this node
        node_size = int(self.graph.nodes[node])
        #node_sequence = self.graph.get_node_sequence(node)
        node_sequence = self.graph.get_node_subsequence(node, offset, min(offset + bases_left, node_size))
        #loggign.info("Got node sequence for node %d, from offset %d to %d: %s" % (node, offset, min(offset + bases_left, node_size), node_sequence))

        # Special case for empty node (without sequence, we always want to add this
        if node_size == 0:
            # add a dummy sequence
            self._bases_in_search_path.append("-")
            self._nodes_in_path.append(node)


        for node_position in range(int(offset), node_size):
            base = node_sequence[node_position-int(offset)]  # subtract offset since node sequence now starts at offset
            #loggign.info("   Adding base %s at node %d, pos %d" % (base, node, node_position))
            self._bases_in_search_path.append(base)
            self._nodes_in_path.append(node)
            bases_left -= 1

            if bases_left == 0:
                # Remove dummy sequence "-" when adding kmer. These are added on indel nodes
                self._add_kmer(''.join(self._bases_in_search_path).replace("-", ""), set(self._nodes_in_path))
                #loggign.info("Recursion is done, added kmer with nodes: %s" % str(self._nodes_in_path))
                # Recursion is done
                return

        # If offset is last base in this node, recursively search next nodes depth first
        next_nodes = self.graph.get_edges(node)
        if len(next_nodes) > 1:
            self._has_traversed_variant = True

        # Prioritize linear ref node
        if len(next_nodes) > 0 and not self.graph.is_linear_ref_node_or_linear_ref_dummy_node(next_nodes[0]):
            next_nodes = list(reversed(next_nodes))

        if len(next_nodes) > 1:
            if self._n_variant_nodes_passed >= self._max_variant_nodes:
                #loggign.info("Passet too many variant nodes, only following linear ref at next nodes %s" % (next_nodes))
                # Only choose linear ref node
                next_nodes = [n for n in next_nodes if self.graph.is_linear_ref_node_or_linear_ref_dummy_node(n)]

        bases_so_far = len(self._bases_in_search_path)
        #loggign.info("Next nodes: %s" % str(next_nodes))
        for next_node in next_nodes:
            self._search_graph_from(next_node, 0, bases_left)
            self._bases_in_search_path = self._bases_in_search_path[0:bases_so_far]
            self._nodes_in_path = self._nodes_in_path[0:bases_so_far]


    def find_kmers_from_linear_ref_position(self, pos):
        self._find_all_variant_kmers_from_position(pos)

    def get_flat_kmers(self, v=None):
        return FlatKmers(np.array(self._hashes, dtype=np.uint64), np.array(self._nodes, np.uint32), np.array(self._ref_offsets, np.uint64), np.array(self._allele_frequencies, np.single))

    def find_kmers_on_linear_reference(self):
        #assert self.spacing == 1, "Finding kmers on linear reference is only possible when spacing is 1"
        #loggign.info("Fetching reference sequence between %d and %d" % (self._start_position, self._end_position+self.k))
        reference_sequence = str(self.reference[self._start_position:self._end_position+self.k])
        assert len(reference_sequence) > 0, "No reference sequence between positions %d and %d" % (self._start_position, self._end_position+self.k)
        #loggign.info("Fetching kmers")
        from .read_kmers import ReadKmers
        kmers = ReadKmers.get_kmers_from_read_dynamic(reference_sequence, power_array(self.k))
        kmers = kmers[::self.spacing]
        #loggign.info("Done fetching kmers")

        self._hashes = kmers
        self._nodes = np.zeros(len(kmers)) + 1
        self._ref_offsets = np.arange(self._start_position, self._start_position+len(reference_sequence), self.spacing)
        self._allele_frequencies = np.zeros(len(kmers)) + 1.0

    def find_kmers(self):
        if self.reference is not None:
            logging.warning("Will find kmers on linear reference and not graph")
            self.find_kmers_on_linear_reference()
            return self.get_flat_kmers()

        #loggign.info("Linear reference is %d bp" % self.graph.linear_ref_length())
        if self._end_position is None:
            self._end_position = self.graph.linear_ref_length()


        prev_time = time.time()
        for i in range(self._start_position // self.spacing, self.graph.linear_ref_length() // self.spacing):
            pos = i * self.spacing
            if i % 50000 == 0:

                logging.info("On ref position %d/%s. Time spent: %.3f. %d/%d basepairs traversed. %d kmers found. Have pruned %d kmers. "
                             "Skipped %d kmers. Skipped due to high frequency: %d. Skipped because too many variant nodes: %d. Skipped because blacklisted node: %d. Skipped low allele freq: %d"
                             % (pos, self._end_position, time.time() - prev_time, (i - self._start_position//self.spacing) * self.spacing, self._end_position-self._start_position,
                                self._kmers_found, self._n_kmers_pruned, self._n_kmers_skipped, self._n_skipped_due_to_frequency, self._n_skipped_due_to_max_variant_nodes, self._n_skipped_blacklist_nodes,
                                self._n_kmers_skipped_low_allele_frequency))

                prev_time = time.time()
                if self._whitelist is not None:
                    logging.info("N skipped because not in whitelist: %d" % self._n_skipped_whitelist)
            if self._end_position is not None and pos >= self._end_position:
                logging.info("Ending at end position %d" % self._end_position)
                break
            self.find_kmers_from_linear_ref_position(pos)


        return self.get_flat_kmers()


if __name__ == "__main__":

    test_simple()
