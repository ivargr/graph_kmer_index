import logging
import time

from .snp_kmer_finder import SnpKmerFinder
from .flat_kmers import FlatKmers
from obgraph import VariantNotFoundException
from .kmer_finder import DenseKmerFinder


class UniqueVariantKmersFinder:
    def __init__(self, graph, variant_to_nodes, variants, k=31, max_variant_nodes=6,
                 kmer_index_with_frequencies=None, haplotype_matrix=None, node_to_variants=None,
                 do_not_choose_lowest_frequency_kmers=False, use_dense_kmer_finder=False, position_id_index=None,
                 use_simple=False):
        self.graph = graph
        self.variant_to_nodes = variant_to_nodes
        self.reference_kmer_index = None
        self.variants = variants
        self.k = k
        self.flat_kmers_found = []
        self.n_failed_variants = 0
        self._n_skipped_because_added_on_other_node = 0
        self._max_variant_nodes = max_variant_nodes
        self._kmer_index_with_frequencies = kmer_index_with_frequencies
        self.haplotype_matrix = haplotype_matrix
        self.node_to_variants = node_to_variants
        self._use_dense_kmer_finder = use_dense_kmer_finder
        self._position_id_index = position_id_index
        self._nodes_found = set()
        self._use_simple = use_simple

        if self._use_dense_kmer_finder:
            assert self._position_id_index is not None, "Position id index must be set when using dense kmer finder"

        self._choose_kmers_with_lowest_frequencies = True
        if do_not_choose_lowest_frequency_kmers:
            self._choose_kmers_with_lowest_frequencies = False

    def kmer_is_unique_on_reference_position(self, kmer, reference_position, ref_start, ref_end):
        # returns true if kmer is unique when ignoring kmers on same ref pos
        reference_kmers = self.reference_kmer_index.get_between(ref_start, ref_end)
        # Remove the one matching our position
        #logging.info("    Checking whether %d among reference kmers: %s" % (kmer, reference_kmers))
        for i, reference_kmer in enumerate(reference_kmers):
            pos = ref_start + i
            if pos != reference_position and reference_kmer == kmer:
                return False
        return True

    def _find_all_kmers_from_position_that_covers_node(self, chromosome, ref_position, cover_node):
        possible_ref_position_adjusted = self.graph.convert_chromosome_ref_offset_to_graph_ref_offset(
            ref_position, chromosome)
        finder = DenseKmerFinder(self.graph, self.k, None, position_id=self._position_id_index,
                                 max_variant_nodes=self._max_variant_nodes, only_store_nodes=set(cover_node),
                                 )
        position_node = self.graph.get_node_at_ref_offset(possible_ref_position_adjusted)
        position_offset = self.graph.get_node_offset_at_ref_offset(possible_ref_position_adjusted)
        finder.find_only_kmers_starting_at_position(node, offset)
        return finder.kmers_found()


    def _find_all_possible_kmer_sets_covering_node(self):
        pass


    def find_kmers_over_variant_node(self, variant, node):
        # tries to find the best possible kmers to represent node
        # if node is small, find only one kmer
        # if node is large, allow for multiple kmers
        # tries to choose kmers thare are rare

        start_pos = variant.position
        assert variant.type is not None, "Variant type must be set"
        if variant.type != "SNP":
            start_pos = start_pos + 1  # start pos for indels is 1 base before

        start_pos -= 1  # go from 1-based to 0-based coordinates

        node_sequence = self.graph.get_node_sequence(node)
        if node_sequence == "" or variant.type != "SNP":
            # empty node, we cannot start at node, start one node before
            # do this for both alleles if indel, so that we start from the same pos
            #logging.info("Finding start node by searching for ref offset %d at chromosomse %d" % (start_pos-1, variant.chromosome))
            start_node = self.graph.get_node_at_chromosome_and_chromosome_offset(variant.chromosome, start_pos-8)
            start_node_offset = self.graph.get_node_offset_at_chromosome_and_chromosome_offset(variant.chromosome, start_pos-8)
            #start_node_offset = self.graph.get_node_size(start_node)-
        else:
            start_node = node
            start_node_offset = 0

        finder = DenseKmerFinder(self.graph, self.k, None, position_id=self._position_id_index,
                                 max_variant_nodes=self._max_variant_nodes, only_store_nodes=set([node]),
                                 only_follow_nodes=set([node])
                                 )
        #logging.info("Searching from node %d offset %d for variant %s" % (start_node, start_node_offset, variant))
        finder.find_only_kmers_starting_at_position(start_node, start_node_offset)
        result = finder.get_flat_kmers(v="1")
        return result


    def find_kmers_over_structural_variant(self, variant, ref_node, variant_node):
        # find over ref node and variant node
        self.find_kmers_over_variant_node(variant, ref_node)
        self.find_kmers_over_variant_node(variant, variant_node)


    def find_kmers_over_variant(self, variant, ref_node, variant_node):
        # more simple version, does not try to find unique kmers
        ref_kmers = self.find_kmers_over_variant_node(variant, ref_node)
        variant_kmers = self.find_kmers_over_variant_node(variant, variant_node)
        return FlatKmers.from_multiple_flat_kmers([ref_kmers, variant_kmers])


    def find_unique_kmers_over_variant(self, variant, ref_node, variant_node):
        is_valid = False
        has_added = False
        # Start searching from before. The last position is probably the best for indels, since the sequence after is offset-ed.
        # We want to end on this, because this is what is chosen if all else fails
        possible_ref_positions = [variant.position - i for i in range(2, self.k - 2)][::4][::-1]
        valid_positions_found = []

        for possible_ref_position in possible_ref_positions:
            possible_ref_position_adjusted = self.graph.convert_chromosome_ref_offset_to_graph_ref_offset(possible_ref_position, variant.chromosome)
            is_valid = True
            only_store_nodes = set()
            for n in (ref_node, variant_node):
                if n not in self._nodes_found:
                    only_store_nodes.add(n)

            if not self._use_dense_kmer_finder:
                finder = SnpKmerFinder(self.graph, self.k, max_variant_nodes=self._max_variant_nodes, only_store_nodes=only_store_nodes,
                                       haplotype_matrix=self.haplotype_matrix, node_to_variants=self.node_to_variants, variant_to_nodes=self.variant_to_nodes)
                finder.find_kmers_from_linear_ref_position(possible_ref_position_adjusted)
            else:
                finder = DenseKmerFinder(self.graph, self.k, None, position_id=self._position_id_index,
                                         max_variant_nodes=self._max_variant_nodes, only_store_nodes=only_store_nodes,
                                         )
                node = self.graph.get_node_at_ref_offset(possible_ref_position_adjusted)
                offset = self.graph.get_node_offset_at_ref_offset(possible_ref_position_adjusted)
                finder.find_only_kmers_starting_at_position(node, offset)


            kmers_ref = set()
            kmers_variant = set()
            for kmer, nodes, ref_position, hash in finder.kmers_found:
                if ref_node in nodes:
                    kmers_ref.add(hash)
                if variant_node in nodes:
                    kmers_variant.add(hash)

                # not a requirement anymore
                #if not self.kmer_is_unique_on_reference_position(hash, ref_position, max(0, ref_position - 150), ref_position + 150 - self.k):
                #    is_valid = False
                #    #logging.info("  Found on linear ref elsewhere, %s" % kmer)
                #    break

                #if hash in self._hashes_added:
                #    is_valid = False
                #    self._n_skipped_because_added_on_other_node += 1

            if not is_valid:
                continue

            # Check if there are identical kmers on the two variant nodes
            if len(kmers_ref.intersection(kmers_variant)) > 0:
                #logging.info("    Felles kmer mellom ref og variant: %s / %s" % (kmers_ref, kmers_variant))
                is_valid = False

            # if we are at last position (then choose this anyway, better than nothing)
            if possible_ref_position == possible_ref_positions[-1]:
                is_valid = True
                #is_valid = False  # testing this

            if is_valid:
                # Kmers are valid, we don't need to search anymore for this variant
                flat = finder.get_flat_kmers(v="1")
                # hack for now: Set ref offset to be the ref offset we searched from
                # new densekmerfinder uses end as ref offset
                #flat._ref_offsets[:] = possible_ref_position_adjusted
                assert len(set(flat._nodes)) <= 2
                valid_positions_found.append(flat)

                if flat.maximum_kmer_frequency(self._kmer_index_with_frequencies) <= 1:
                    # no need to search for more positions, all kemrs are unique
                    break

                if len(flat._nodes) == 0:
                    logging.warning("Found 0 nodes for variant %s. Hashes: %s, ref positions: %s. Searched from ref position %d" % (variant, flat._hashes, flat._ref_offsets, possible_ref_position))
                    #raise Exception("No kmers found")

                if ref_node not in flat._nodes and ref_node in only_store_nodes:
                    logging.warning("Found kmers for variant %s with ref/variant nodes %d/%d but flat kmers does not contain ref node. Flat kmer nodes: %s. Searched from possition %d" % (variant, ref_node, variant_node, flat._nodes, possible_ref_position))
                    #raise Exception("No kmers found")

                if variant_node not in flat._nodes and variant_node in only_store_nodes:
                    logging.warning("No variant node kmers found for variant %s with variant node %d and ref node %d" %
                                    (variant, variant_node, ref_node))
                    logging.warning("Found no variant node kmers for variant %s. Hashes: %s, ref positions: %s. Searched from ref position %d" %
                                    (variant, flat._hashes, flat._ref_offsets, possible_ref_position))
                    logging.warning("Found nodes: %s" % flat._nodes)
                    logging.warning("Only store nodes: %s" % only_store_nodes)
                    logging.warning("Searched from node/offset: %d/%d" % (
                        self.graph.get_node_at_ref_offset(possible_ref_position),
                        self.graph.get_node_offset_at_ref_offset(possible_ref_position)
                    ))
                    #raise Exception("Errror")

        # Sort positions by max kmer frequency
        if len(valid_positions_found) == 0:
            logging.warning("Found no positions with valid kmers for variant %s" % variant)
            self.n_failed_variants += 1
            return

        if self._choose_kmers_with_lowest_frequencies:
            valid_positions_found = sorted(valid_positions_found, key=lambda p: p.maximum_kmer_frequency(self._kmer_index_with_frequencies))

        #valid_positions_found = sorted(valid_positions_found, key=lambda p: p.sum_of_kmer_frequencies(self._kmer_index_with_frequencies))
        best_position = valid_positions_found[0]
        nodes_found_here = list(best_position._nodes)
        if nodes_found_here.count(ref_node) > 1000 and False:
            logging.warning("Ref node %d found %d times" % (ref_node, nodes_found_here.count(ref_node)))
            logging.warning("Starting positions: %s" % possible_ref_positions)
            logging.info("Nodes: %s" % best_position._nodes)
            logging.info("Variant: %s" % variant)
            logging.info("Nodse from all positions:")
            for pos in valid_positions_found:
                logging.info(pos._nodes)

            assert False


        for node in set(best_position._nodes):
            assert node not in self._nodes_found, "Found node %d at variant %s (nodes %d/%d), but already found for previous variant" % \
                                                  (node, variant, ref_node, variant_node)
            self._nodes_found.add(node)

        return best_position
        #self.flat_kmers_found.append(best_position)


    def find_unique_kmers(self):
        prev_time = time.time()
        for i, variant in enumerate(self.variants):
            n_processed = len(self.flat_kmers_found)
            if i % 5000 == 0:
                logging.info("%d/%d variants processed (time spent on previous 5000 variants: %.3f s). "
                             "Now on chromosome/ref pos %s/%d" % (i, len(self.variants), time.time()-prev_time, str(variant.chromosome), variant.position))
                prev_time = time.time()

            #ref_node, variant_node = self.graph.get_variant_nodes(variant)
            assert variant.vcf_line_number is not None, "Variant line number must be specified"
            ref_node = self.variant_to_nodes.ref_nodes[variant.vcf_line_number]
            variant_node = self.variant_to_nodes.var_nodes[variant.vcf_line_number]

            if ref_node == 0 or variant_node == 0:
                continue

            if not self._use_simple:
                self.flat_kmers_found.append(self.find_unique_kmers_over_variant(variant, ref_node, variant_node))
            else:
                self.flat_kmers_found.append(self.find_kmers_over_variant(variant, ref_node, variant_node))

            if len(self.flat_kmers_found) != n_processed + 1:
                logging.warning("DID NOT FIND KMERS ON %s" % variant)

        logging.info("N variants with kmers found: %d" % len(self.flat_kmers_found))
        logging.info("Done with all variants. N that failed: %d" % self.n_failed_variants)

        return FlatKmers.from_multiple_flat_kmers(self.flat_kmers_found)

