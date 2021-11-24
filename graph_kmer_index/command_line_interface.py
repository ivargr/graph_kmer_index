import sys
import argparse
import logging
#from multiprocessing import shared_memory
from multiprocessing import Pool, Process
import numpy as np
from itertools import repeat
from pyfaidx import Fasta

from .collision_free_kmer_index import CollisionFreeKmerIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
from obgraph import Graph
from .snp_kmer_finder import SnpKmerFinder
import pickle
from .flat_kmers import FlatKmers
from .reverse_kmer_index import ReverseKmerIndex
from .unique_kmer_index import UniqueKmerIndex
from .reference_kmer_index import ReferenceKmerIndex
from pathos.multiprocessing import Pool
from obgraph.variants import VcfVariants
from .unique_variant_kmers import UniqueVariantKmersFinder
from graph_kmer_index.shared_mem import to_shared_memory, from_shared_memory, remove_shared_memory_in_session
from obgraph.variant_to_nodes import VariantToNodes, NodeToVariants
from obgraph.haplotype_matrix import HaplotypeMatrix


def main():
    run_argument_parser(sys.argv[1:])


def create_index_single_thread(args, interval=None):
    start_position = None
    end_position = None
    if interval is not None:
        start_position = interval[0]
        end_position = interval[1]

    logging.info("Loading data")
    #graph = Graph.from_file(args.graph_file_name)
    if args.graph_file_name is not None:
        graph = from_shared_memory(Graph, "graph_shared"+args.shared_memory_unique_id)
        reference = None
    else:
        graph = None
        assert args.reference_fasta is not None
        assert args.reference_name is not None, "Reference name must be specified"
        logging.info("Reference name is *%s*" % args.reference_name)
        try:
            fasta = Fasta(args.reference_fasta)
            logging.info("Names in fasta: %s" % str(fasta.keys()))
            reference = fasta[args.reference_name]
        except KeyError:
            logging.error("Did not find reference name %s in %s" % (args.reference_name, args.reference_fasta))
            raise

    logging.info("Running kmerfinder")
    whitelist = None
    if args.whitelist is not None:
        w = FlatKmers.from_file(args.whitelist)
        whitelist = set(w._hashes)

    skip_kmers_with_nodes = None
    if args.skip_kmers_with_nodes is not None:
        f = FlatKmers.from_file(args.skip_kmers_with_nodes)
        skip_kmers_with_nodes = set(f._nodes)

    finder = SnpKmerFinder(graph, k=args.kmer_size, spacing=args.spacing,
                           include_reverse_complements=args.include_reverse_complement,
                           pruning=args.pruning,
                           max_kmers_same_position=args.max_kmers_same_position,
                           max_frequency=args.max_frequency,
                           max_variant_nodes=args.max_variant_nodes,
                           only_add_variant_kmers=args.only_add_variant_kmers,
                           whitelist=whitelist,
                           only_save_variant_nodes=args.only_save_variant_nodes,
                           start_position=start_position,
                           end_position=end_position,
                           skip_kmers_with_nodes=skip_kmers_with_nodes,
                           only_save_one_node_per_kmer=args.only_save_one_node_per_kmer,
                           reference=reference)

    kmers = finder.find_kmers()
    return kmers

def create_index(args):
    args.shared_memory_unique_id = str(np.random.randint(0, 10e15))
    r = args.shared_memory_unique_id

    if args.graph_file_name is not None:
        graph = Graph.from_file(args.graph_file_name)
        to_shared_memory(graph, "graph_shared"+r)

    if args.threads == 1:
        kmers = create_index_single_thread(args)
        kmers.to_file(args.out_file_name)
    else:
        logging.info("Making pool with %d workers" % args.threads)
        pool = Pool(args.threads)
        genome_size = args.genome_size
        n_total_start_positions = genome_size // args.spacing
        n_positions_each_process = n_total_start_positions // args.threads
        logging.info("Using genome size %d. Will process %d genome positions in each process." % (genome_size, n_positions_each_process))
        intervals = []
        for i in range(args.threads):
            start_position = n_positions_each_process * i * args.spacing
            end_position = n_positions_each_process * (i+1) * args.spacing
            intervals.append((start_position, end_position))
            logging.info("Creating interval for genome segment %d-%d" % (start_position, end_position))

        all_hashes = []
        all_nodes = []
        all_ref_offsets = []
        all_allele_frequencies = []
        for flat_kmers in pool.starmap(create_index_single_thread, zip(repeat(args), intervals)):
            all_hashes.append(flat_kmers._hashes)
            all_nodes.append(flat_kmers._nodes)
            all_ref_offsets.append(flat_kmers._ref_offsets)
            all_allele_frequencies.append(flat_kmers._allele_frequencies)

        logging.info("Making full index from all indexes")
        full_index = FlatKmers(
            np.concatenate(all_hashes),
            np.concatenate(all_nodes),
            np.concatenate(all_ref_offsets),
            np.concatenate(all_allele_frequencies)
        )

        logging.info("Saving full index")
        full_index.to_file(args.out_file_name)


def make_from_flat(args):
    flat = FlatKmers.from_file(args.flat_index)
    index = CollisionFreeKmerIndex.from_flat_kmers(flat, modulo=args.hash_modulo, skip_frequencies=args.skip_frequencies,
                                                   skip_singletons=args.skip_singletons)
    index.to_file(args.out_file_name)
    logging.info("Done making kmer index")


def make_reverse(args):
    flat = FlatKmers.from_file(args.flat_index)
    reverse = ReverseKmerIndex.from_flat_kmers(flat)
    reverse.to_file(args.out_file_name)
    logging.info("Done. Wrote reverse index to file: %s" % args.out_file_name)


def make_reference_kmer_index(args):
    if args.reference_fasta is not None:
        logging.info("Making from a linear reference")
        index = ReferenceKmerIndex.from_linear_reference(args.reference_fasta, args.reference_name, args.kmer_size, args.only_store_kmers)
    else:
        flat = FlatKmers.from_file(args.flat_index)
        index = ReferenceKmerIndex.from_flat_kmers(flat)

    index.to_file(args.out_file_name)
    logging.info("Saved reference kmer index to file %s" % args.out_file_name)


def make_unique_index(args):
    graph = Graph.from_file(args.graph)
    reverse = ReverseKmerIndex.from_file(args.reverse)
    flat = FlatKmers.from_file(args.flat_index)
    unique = UniqueKmerIndex.from_flat_kmers_and_snps_graph(flat, graph, reverse)
    unique.to_file(args.out_file_name)


def prune_flat_kmers(args):
    index = FlatKmers.from_file(args.flat_index)
    new_hashes = []
    new_nodes = []
    new_ref_offsets = []

    prev_hash = -1
    prev_ref_offset = -1
    n_skipped = 0
    for i in range(len(index._hashes)):
        if i % 1000000 == 0:
            logging.info("%d processed, %d skipped" % (i, n_skipped))

        if index._hashes[i] == prev_hash and index._ref_offsets[i] == prev_ref_offset:
            n_skipped += 1
            continue


        new_hashes.append(index._hashes[i])
        new_nodes.append(index._nodes[i])
        new_ref_offsets.append(index._ref_offsets[i])

        prev_hash = index._hashes[i]
        prev_ref_offset = index._ref_offsets[i]

    new = FlatKmers(
        np.array(new_hashes, dtype=index._hashes.dtype),
        np.array(new_nodes, dtype=index._nodes.dtype),
        np.array(new_ref_offsets, dtype=index._ref_offsets.dtype),
    )




def run_argument_parser(args):
    parser = argparse.ArgumentParser(
        description='Graph Kmer Index.',
        prog='graph_kmer_index',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=100))

    subparsers = parser.add_subparsers()
    subparser = subparsers.add_parser("make")
    subparser.add_argument("-g", "--graph_file_name", required=False)
    subparser.add_argument("-o", "--out_file_name", required=True)
    subparser.add_argument("-k", "--kmer_size", required=False, type=int, default=31)
    subparser.add_argument("-r", "--include-reverse-complement", required=False, type=bool, default=False)
    subparser.add_argument("-s", "--spacing", required=False, type=int, default=31)
    subparser.add_argument("-p", "--pruning", required=False, type=bool, default=False, help="Set to True to prune unecessary kmers")
    subparser.add_argument("-m", "--max-kmers-same-position", required=False, type=int, default=100000, help="Maximum number of kmers allowd to be added from the same ref position")
    subparser.add_argument("-M", "--max-frequency", required=False, type=int, default=10000000, help="Skip kmers with frequency higher than this. Will never skip kmers crossing variants.")
    subparser.add_argument("-v", "--max-variant-nodes", required=False, type=int, default=100000, help="Max variant nodes allowed in kmer.")
    subparser.add_argument("-V", "--only-add-variant-kmers", required=False, type=bool, default=False)
    subparser.add_argument("-N", "--only-save-variant-nodes", required=False, type=bool, default=False)
    subparser.add_argument("-O", "--only-save-one-node-per-kmer", required=False, type=bool, default=False)
    subparser.add_argument("-S", "--skip-kmers-with-nodes", required=False, help="Skip kmers with nodes that exist in this flat kmers file")
    subparser.add_argument("-w", "--whitelist", required=False, help="Only add kmers in this whitelist (should be a flat kmers file)")
    subparser.add_argument("-t", "--threads", required=False, default=1, type=int, help="How many threads to use. Some parameters will have local effect if t > 1 (-M)")
    subparser.add_argument("-G", "--genome-size", required=False, default=3000000000, type=int, help="Must be set if --threads > 1 (used to make chunks to run in parallel)")
    subparser.add_argument("-R", "--reference-fasta", required=False, help="Make from this reference fasta instead of graph")
    subparser.add_argument("-n", "--reference-name", required=False, help="Name of reference in fasta file. Needed when reference fasta is used.")

    subparser.set_defaults(func=create_index)

    subparser = subparsers.add_parser("make_from_flat")
    subparser.add_argument("-o", "--out_file_name", required=True)
    subparser.add_argument("-f", "--flat-index", required=True)
    subparser.add_argument("-m", "--hash_modulo", required=False, type=int, default=452930477)
    subparser.add_argument("-S", "--skip-frequencies", type=bool, default=False, required=False)
    subparser.add_argument("-s", "--skip-singletons", type=bool, default=False, required=False)
    subparser.set_defaults(func=make_from_flat)

    subparser = subparsers.add_parser("make_reverse")
    subparser.add_argument("-f", "--flat-index", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.set_defaults(func=make_reverse)

    subparser = subparsers.add_parser("make_unique_index", help="Make index from unique kmers in the graph to nodes that are covered by those kmers")
    subparser.add_argument("-f", "--flat-index", required=True)
    subparser.add_argument("-g", "--graph", required=True)
    subparser.add_argument("-r", "--reverse", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.set_defaults(func=make_unique_index)

    subparser = subparsers.add_parser("make_reference_kmer_index", help="Make index that allows lookup between two ref positions")
    subparser.add_argument("-f", "--flat-index", required=False, help="If set, will create from flat kmers")
    subparser.add_argument("-r", "--reference-fasta", required=False, help="If set, will create from a linear reference")
    subparser.add_argument("-n", "--reference-name", required=False, help="Only needed if creating from linear reference")
    subparser.add_argument("-k", "--kmer-size", required=False, type=int, default=16, help="Only needed if making from linear fasta")
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument("-O", "--only-store-kmers", required=False, default=False, type=bool, help="Can be used when making from fasta file, will then not store the index since an index is not needed")
    subparser.set_defaults(func=make_reference_kmer_index)

    def make_unique_variant_kmers_single_thread(variants, args):
        r = args.shared_memory_unique_id
        variant_to_nodes = from_shared_memory(VariantToNodes, "variant_to_nodes_shared"+r)
        kmer_index = from_shared_memory(CollisionFreeKmerIndex, "kmer_index_shared"+r)
        graph = from_shared_memory(Graph, "graph_shared"+r)
        #graph = Graph.from_file(args.graph)
        logging.info("Reading all variants")

        node_to_variants = None
        haplotype_matrix = None
        if args.haplotype_matrix is not None:
            haplotype_matrix = HaplotypeMatrix.from_file(args.haplotype_matrix)
            node_to_variants = NodeToVariants.from_file(args.node_to_variants)

        finder = UniqueVariantKmersFinder(graph, variant_to_nodes, variants, args.kmer_size, args.max_variant_nodes,
                                          kmer_index_with_frequencies=kmer_index, haplotype_matrix=haplotype_matrix,
                                          node_to_variants=node_to_variants,
                                          do_not_choose_lowest_frequency_kmers=args.do_not_choose_lowest_frequency_kmers)
        flat_kmers = finder.find_unique_kmers()
        return flat_kmers

    def make_unique_variant_kmers(args):
        args.shared_memory_unique_id = str(np.random.randint(0,10e15))
        r = args.shared_memory_unique_id
        logging.info("Reading kmer index")
        kmer_index = CollisionFreeKmerIndex.from_file(args.kmer_index)
        to_shared_memory(kmer_index, "kmer_index_shared"+r)
        logging.info("Reading variant to nodes")
        variant_to_nodes = VariantToNodes.from_file(args.variant_to_nodes)
        to_shared_memory(variant_to_nodes, "variant_to_nodes_shared"+r)
        logging.info("REading graph")
        graph = Graph.from_file(args.graph)
        to_shared_memory(graph, "graph_shared"+r)
        logging.info("Reading all variants")
        variants = VcfVariants.from_vcf(args.vcf, skip_index=True, make_generator=True)
        variants = variants.get_chunks(chunk_size=args.chunk_size)
        pool = Pool(args.n_threads)

        all_flat_kmers = []
        for flat_kmers in pool.starmap(make_unique_variant_kmers_single_thread, zip(variants, repeat(args))):
            all_flat_kmers.append(flat_kmers)

        logging.info("Merge all flat kmers")
        merged_flat = FlatKmers.from_multiple_flat_kmers(all_flat_kmers)
        merged_flat.to_file(args.out_file_name)
        logging.info("Wrote to file %s" % args.out_file_name)

    subparser = subparsers.add_parser("make_unique_variant_kmers", help="Make a reverse variant index lookup to unique kmers on that variant")
    subparser.add_argument("-g", "--graph", required=True)
    subparser.add_argument("-V", "--variant_to_nodes", required=True)
    subparser.add_argument("-N", "--node-to-variants", required=False)
    subparser.add_argument("-H", "--haplotype-matrix", required=False)
    subparser.add_argument("-k", "--kmer-size", required=True, type=int)
    subparser.add_argument("-i", "--kmer-index", required=True, help="Kmer index used to check frequency of kmers in genome")
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument("-v", "--vcf", required=True)
    subparser.add_argument("-t", "--n-threads", required=False, default=1, type=int)
    subparser.add_argument("-c", "--chunk-size", required=False, default=10000, type=int, help="Number of variants given to each thread")
    subparser.add_argument("-m", "--max-variant-nodes", required=False, default=6, type=int, help="Maximum number of variant nodes allowed in kmer")
    subparser.add_argument("-d", "--do-not-choose-lowest-frequency-kmers", required=False, type=bool, help="For testing only. Will not choose the best kmers.")
    subparser.set_defaults(func=make_unique_variant_kmers)

    subparser = subparsers.add_parser("prune_flat_kmers")
    subparser.add_argument("-f", "--flat-index", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.set_defaults(func=prune_flat_kmers)

    def merge_flat_kmers(args):
        new = FlatKmers.from_multiple_flat_kmers([FlatKmers.from_file(f) for f in args.flat_kmers.split(",")])
        new.to_file(args.out_file_name)
        logging.info("Wrote merged index to %s" % new)

    subparser = subparsers.add_parser("merge_flat_kmers")
    subparser.add_argument("-f", "--flat-kmers", required="true", help="Comma-separeted list of file names to be merged")
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.set_defaults(func=merge_flat_kmers)

    def make_kmer_frequencies(args):
        from .kmer_frequency_index import KmerFrequencyIndex
        ref_kmers = ReferenceKmerIndex.from_file(args.reference_kmers)
        index = KmerFrequencyIndex.from_kmers(ref_kmers.kmers)
        index.to_file(args.out_file_name)
        logging.info("Wrote to file %s" % args.out_file_name)

    subparser = subparsers.add_parser("make_kmer_frequency_index")
    subparser.add_argument("-r", "--reference-kmers", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.set_defaults(func=make_kmer_frequencies)

    def set_frequencies_using_other_index(args):

        logging.info("Reading index")
        index = CollisionFreeKmerIndex.from_file(args.kmer_index)
        logging.info("Reading other index")
        other = CollisionFreeKmerIndex.from_file(args.kmer_index_with_frequencies)
        index.set_frequencies_using_other_index(other, args.multiplier)
        index.to_file(args.kmer_index)
        logging.info("Wrote index to file %s" % args.kmer_index)

    subparser = subparsers.add_parser("set_frequencies_using_other_index")
    subparser.add_argument("-i", "--kmer-index", required=True)
    subparser.add_argument("-f", "--kmer-index-with-frequencies", required=True)
    subparser.add_argument("-m", "--multiplier", required=False, type=int, default=1)
    subparser.set_defaults(func=set_frequencies_using_other_index)

    def set_allele_frequencies(args):
        index = CollisionFreeKmerIndex.from_file(args.kmer_index)
        frequencies = np.load(args.frequencies)
        index.set_allele_frequencies(frequencies)
        index.to_file(args.kmer_index)
        logging.info("Wrote index to file %s" % args.kmer_index)

    subparser = subparsers.add_parser("set_allele_frequencies")
    subparser.add_argument("-i", "--kmer-index", required=True)
    subparser.add_argument("-f", "--frequencies", required=True)
    subparser.set_defaults(func=set_allele_frequencies)

    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)
    remove_shared_memory_in_session()

