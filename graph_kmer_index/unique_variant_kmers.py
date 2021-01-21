import logging
from .snp_kmer_finder import SnpKmerFinder
from .flat_kmers import FlatKmers
from obgraph import VariantNotFoundException

class UniqueVariantKmersFinder:
    def __init__(self, graph, variants, k=31):
        self.graph = graph
        self.reference_kmer_index = None
        self.variants = variants
        self.k = k
        self.flat_kmers_found = []
        self.n_failed_variants = 0
        self._n_skipped_because_added_on_other_node = 0
        self._hashes_added = set()

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

    def find_unique_kmers_over_variant(self, variant, ref_node, variant_node):
        is_valid = False
        has_added = False
        # Start searching from before. The last position is probably the best for indels, since the sequence after is offset-ed. We want to end on this, because this is what is chosen if all else fails
        possible_ref_positions = [variant.position - i for i in range(1, self.k - 2)][::-1]
        for possible_ref_position in possible_ref_positions:

            is_valid = True
            finder = SnpKmerFinder(self.graph, self.k, max_variant_nodes=6, only_store_nodes=set([ref_node, variant_node]))
            finder.find_kmers_from_linear_ref_position(possible_ref_position)


            kmers_ref = set()
            kmers_variant = set()
            for kmer, nodes, ref_position, hash in finder.kmers_found:
                if ref_node in nodes:
                    kmers_ref.add(kmer)
                if variant_node in nodes:
                    kmers_variant.add(kmer)

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

            if is_valid:
                # Kmers are valid, we don't need to search anymore for this variant
                flat = finder.get_flat_kmers()
                if len(flat._nodes) == 0:
                    logging.warning("Found 0 nodes for variant %s. Hashes: %s, ref positions: %s" % (variant, flat._hashes, flat._ref_offsets))

                if ref_node not in flat._nodes:
                    logging.warning("Found kmers for variant %s with ref/variant nodes %d/%d but flat kmers does not contain ref node. Flat kmer nodes: %s" % (variant, ref_node, variant_node, flat._nodes))

                if variant_node not in flat._nodes:
                    logging.warning("No variant node kmers found for variant %s" % variant)

                self.flat_kmers_found.append(flat)
                has_added = True
                for hash in flat._hashes:
                    self._hashes_added.add(hash)


                break

        if not has_added:
            logging.warning("Did not add any kmers for variant %s (nodes: %d/%d)" % (variant, ref_node, variant_node))
        if not is_valid:
            logging.warning("Traversed a variant %s, nodes %d/%d without finding unique kmers" % (variant, ref_node, variant_node))
            self.n_failed_variants += 1

    def find_unique_kmers(self):
        for i, variant in enumerate(self.variants):
            n_processed = len(self.flat_kmers_found)
            if i % 10000 == 0:
                logging.info("%d variants processed. Now on ref pos %d" % (i, variant.position))

            try:
                ref_node, variant_node = self.graph.get_variant_nodes(variant)
            except VariantNotFoundException:
                logging.warning("Variant %s not found" % variant)
                continue

            self.find_unique_kmers_over_variant(variant, ref_node, variant_node)

            if len(self.flat_kmers_found) != n_processed + 1:
                logging.warning("DID NOT FIND KMERS ON %s" % variant)

        logging.info("N variants with kmers found: %d" % len(self.flat_kmers_found))
        logging.info("Done with all variants. N that failed: %d" % self.n_failed_variants)

        return FlatKmers.from_multiple_flat_kmers(self.flat_kmers_found)

