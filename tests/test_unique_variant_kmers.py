from obgraph import Graph
from alignment_free_graph_genotyper.analysis import GenotypeCalls, VariantGenotype
from graph_kmer_index.unique_variant_kmers import UniqueVariantKmersFinder
from graph_kmer_index.reference_kmer_index import ReferenceKmerIndex

def simple_test():
    g = Graph.from_dicts(
        {
            1: "CTACCA",
            2: "AA",
            3: "TAAATAA",
            4: ""
        },
        {
            1: [2, 4],
            2: [3],
            4: [3]

        },
        [1, 2, 3]
    )
    print(g.ref_offset_to_node)
    print(g.get_node_size(3))
    k = 4
    variants = GenotypeCalls([VariantGenotype(6, "AAA", "A", "", "DELETION")])
    reference_kmers = ReferenceKmerIndex.from_sequence("CTACCAAATAAATAA", k)
    finder = UniqueVariantKmersFinder(g, reference_kmers, variants, k)
    finder.find_unique_kmers()


simple_test()
