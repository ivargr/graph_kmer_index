from .snp_kmer_finder import SnpKmerFinder

class IndelKmerFinder(SnpKmerFinder):


    def __init__(self, graph, k=15, spacing=None, include_reverse_complements=False, pruning=False, max_kmers_same_position=100000,
                 max_frequency=10000, max_variant_nodes=10000, only_add_variant_kmers=False, whitelist=None, only_save_variant_nodes=False,
                 start_position=None, end_position=None):

        super(IndelKmerFinder, self).__init__(graph, k, spacing, include_reverse_complements, pruning, max_kmers_same_position,
                 max_frequency, max_variant_nodes, only_add_variant_kmers, whitelist, only_save_variant_nodes,
                 start_position, end_position)


