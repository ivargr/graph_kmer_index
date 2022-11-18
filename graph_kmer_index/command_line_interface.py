import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
import sys
import argparse
import itertools
#from multiprocessing import shared_memory
import time
from multiprocessing import Pool, Process
import numpy as np
from itertools import repeat
from pyfaidx import Fasta

from .collision_free_kmer_index import CollisionFreeKmerIndex

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
from shared_memory_wrapper.shared_memory import to_shared_memory, from_shared_memory, remove_shared_memory_in_session, to_file, from_file, object_to_shared_memory, object_from_shared_memory
from shared_memory_wrapper import get_shared_pool, close_shared_pool
from obgraph.variant_to_nodes import VariantToNodes, NodeToVariants
from obgraph.haplotype_matrix import HaplotypeMatrix
from npstructures import HashTable, Counter
from .npset import NpSet
from .collision_free_kmer_index import CounterKmerIndex
from .kmer_finder import DenseKmerFinder
from .collision_free_kmer_index import KmerIndex2
from .critical_graph_paths import CriticalGraphPaths
from shared_memory_wrapper import to_file, from_file
from obgraph.position_id import PositionId



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
            assert len(reference) > 0, "Length of ref sequennce is 0. Seomthing is wrong"
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
                           include_reverse_complements=False,  # always false, adds revcomp in the end if chosen
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

    if args.include_reverse_complement:
        logging.info("Adding reverse complements")
        kmers_revcomp = kmers.get_reverse_complement_flat_kmers(args.kmer_size)
        kmers = FlatKmers.from_multiple_flat_kmers([kmers, kmers_revcomp])

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
        n_jobs = args.threads * 10
        logging.info("Making pool with %d workers" % n_jobs)
        pool = Pool(args.threads)
        genome_size = args.genome_size
        n_total_start_positions = genome_size // args.spacing
        n_positions_each_process = n_total_start_positions // n_jobs
        logging.info("Using genome size %d. Will process %d genome positions in each process." % (genome_size, n_positions_each_process))
        intervals = []
        for i in range(n_jobs):
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

    if args.add_reverse_complements:
        logging.info("Will add reverse complements of every hash. Getting reverse complements.")
        logging.info("NB: Using k=%d when computing reverse complements" % args.kmer_size)
        rev_comp_flat = flat.get_reverse_complement_flat_kmers(k=args.kmer_size)
        flat = FlatKmers.from_multiple_flat_kmers([flat, rev_comp_flat])
        logging.info("Made new flat kmers with rev comp, now making index")

    if args.make_minimal:
        logging.info("Making minimal kmer index")
        from .collision_free_kmer_index import MinimalKmerIndex
        index = MinimalKmerIndex.from_flat_kmers(flat, modulo=args.hash_modulo)
    else:
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
    subparser.add_argument("-M", "--make-minimal", type=bool, default=False, required=False)
    subparser.add_argument("-r", "--add-reverse-complements", type=bool, default=False, required=False, help="If True, the rev. comp. of every kmer hash will also be added")
    subparser.add_argument("-k", "--kmer-size", type=int, default=31, required=False, help="Only required when --add-reverse-complements=True. Used to compute rev comp hashes.")
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
        args = object_from_shared_memory(args)
        #variant_to_nodes = from_shared_memory(VariantToNodes, "variant_to_nodes_shared"+r)
        variant_to_nodes = args["variant_to_nodes"]

        #kmer_index = from_shared_memory(CollisionFreeKmerIndex, "kmer_index_shared"+r)
        kmer_index = args["kmer_index"]
        if kmer_index is None:
            kmer_index = args["kmer_counter"]

        #graph = from_shared_memory(Graph, "graph_shared"+r)
        graph = args["graph"]

        #graph = Graph.from_file(args.graph)
        logging.info("Reading all variants")

        node_to_variants = None
        haplotype_matrix = None
        """
        if args.haplotype_matrix is not None:
            haplotype_matrix = HaplotypeMatrix.from_file(args.haplotype_matrix)
            node_to_variants = NodeToVariants.from_file(args.node_to_variants)
        """

        use_dense_kmer_finder = False
        if args["use_dense_kmer_finder"]:
            use_dense_kmer_finder = True
            logging.info("Will use dense kmer finder")

        finder = UniqueVariantKmersFinder(graph, variant_to_nodes, variants, args["kmer_size"], args["max_variant_nodes"],
                                          kmer_index_with_frequencies=kmer_index, haplotype_matrix=haplotype_matrix,
                                          node_to_variants=node_to_variants,
                                          do_not_choose_lowest_frequency_kmers=args["do_not_choose_lowest_frequency_kmers"],
                                          use_dense_kmer_finder=use_dense_kmer_finder,
                                          position_id_index=args["position_id_index"],
                                          use_simple=args["simple"]
                                          )
        flat_kmers = finder.find_unique_kmers()
        return flat_kmers

    def make_unique_variant_kmers(args):
        args = vars(args)
        args_orig = args
        args.pop("func")  # necessary for putting args in shared memory

        logging.info("Reading all variants")
        variants = VcfVariants.from_vcf(args["vcf"], skip_index=True, make_generator=True)
        variants = variants.get_chunks(chunk_size=args["chunk_size"])
        pool = Pool(args["n_threads"])

        all_flat_kmers = []
        args = object_to_shared_memory(args)
        for flat_kmers in pool.starmap(make_unique_variant_kmers_single_thread, zip(variants, repeat(args))):
            all_flat_kmers.append(flat_kmers)

        logging.info("Merge all flat kmers")
        merged_flat = FlatKmers.from_multiple_flat_kmers(all_flat_kmers)
        merged_flat.to_file(args_orig["out_file_name"])
        logging.info("Wrote to file %s" % args_orig["out_file_name"])

    subparser = subparsers.add_parser("make_unique_variant_kmers", help="Make a reverse variant index lookup to unique kmers on that variant")
    subparser.add_argument("-g", "--graph", required=True, type=Graph.from_file)
    subparser.add_argument("-V", "--variant_to_nodes", required=True, type=VariantToNodes.from_file)
    subparser.add_argument("-N", "--node-to-variants", required=False)
    subparser.add_argument("-H", "--haplotype-matrix", required=False)
    subparser.add_argument("-k", "--kmer-size", required=True, type=int)
    subparser.add_argument("-i", "--kmer-index", required=False, help="Kmer index used to check frequency of kmers in genome", type=CollisionFreeKmerIndex.from_file)
    subparser.add_argument("-I", "--kmer-counter", required=False, help="Kmer index used to check frequency of kmers in genome", type=from_file)
    subparser.add_argument("-p", "--position-id-index", required=False, type=from_file)
    subparser.add_argument("-D", "--use-dense-kmer-finder", required=False, type=bool, default=False)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument("-v", "--vcf", required=True)
    subparser.add_argument("-t", "--n-threads", required=False, default=1, type=int)
    subparser.add_argument("-c", "--chunk-size", required=False, default=10000, type=int, help="Number of variants given to each thread")
    subparser.add_argument("-m", "--max-variant-nodes", required=False, default=6, type=int, help="Maximum number of variant nodes allowed in kmer")
    subparser.add_argument("-d", "--do-not-choose-lowest-frequency-kmers", required=False, type=bool, help="For testing only. Will not choose the best kmers.")
    subparser.add_argument("-S", "--simple", type=bool, default=False, help="Set to True to use simple kmer selection")
    subparser.set_defaults(func=make_unique_variant_kmers)


    def sample_kmers_from_structural_variants_command(args):
        from .structural_variants import sample_kmers_from_structural_variants
        kmers = sample_kmers_from_structural_variants(args.graph,
                                                      args.variant_to_nodes,
                                                      args.kmer_index,
                                                      args.kmer_size)
        kmers.to_file(args.out_file_name)


    subparser = subparsers.add_parser("sample_kmers_from_structural_variants", help="Samples extra kmers from large structural variant nodes. Meant ot be combined with other variant kmers")
    subparser.add_argument("-g", "--graph", required=True, type=Graph.from_file)
    subparser.add_argument("-V", "--variant_to_nodes", required=True, type=VariantToNodes.from_file)
    subparser.add_argument("-k", "--kmer-size", required=True, type=int)
    subparser.add_argument("-i", "--kmer-index", required=False,
                           help="Kmer index used to check frequency of kmers in genome",
                           type=from_file)
    subparser.add_argument("-I", "--kmer-counter", required=False,
                           help="Kmer index used to check frequency of kmers in genome", type=from_file)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument("-t", "--n-threads", required=False, default=1, type=int)
    subparser.set_defaults(func=sample_kmers_from_structural_variants_command)


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


    def create_counter_index(args):
        kmer_index = CollisionFreeKmerIndex.from_file(args.kmer_index)
        counter_index = CounterKmerIndex.from_kmer_index(kmer_index, args.modulo)
        to_file(counter_index, args.out_file_name)


    subparser = subparsers.add_parser("create_counter_index")
    subparser.add_argument("-i", "--kmer-index", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument("-m", "--modulo", required=False, default=200000033, type=int)
    subparser.set_defaults(func=create_counter_index)


    def index_single_thread(data):
        args, chunk = data
        args = object_from_shared_memory(args)
        logging.info("Processing chunk %s" % str(chunk))

        t = time.perf_counter()
        kmer_finder = DenseKmerFinder(args["graph"], args["kmer_size"], critical_graph_paths=args["critical_graph_paths"],
                                      position_id=args["position_id"],
                                      max_variant_nodes=args["max_variant_nodes"],
                                      only_save_one_node_per_kmer=True,
                                      start_at_critical_path_number=chunk[0],
                                      stop_at_critical_path_number=chunk[1],
                                      whitelist=args["whitelist"])
        kmer_finder.find()
        logging.info("Took %d sec to find kmers" % (time.perf_counter()-t))
        res = kmer_finder.get_flat_kmers(v="1")
        if len(np.where(res._hashes == 979349781162838629)[0]) > 0:
            logging.info("FOUND MATCH! %d/%d" % (chunk[0], chunk[1]))

        return res

    def index(args):
        pool = get_shared_pool(args.n_threads)

        if args.critical_graph_paths is None:
            logging.info("Making critical graph paths since not specified")
            args.critical_paths = CriticalGraphPaths.from_graph(graph, args.k)

        args.position_id = PositionId.from_graph(args.graph) if args.position_id is not None else None

        args = vars(args)
        args.pop("func")
        args_shared = object_to_shared_memory(args)
        critical_paths = args["critical_graph_paths"]

        n_chunks = args["n_threads"]*20
        if n_chunks >= len(critical_paths):
            n_chunks = len(critical_paths)

        assert n_chunks <= len(critical_paths), "Too many threads for data. Try with fewer"

        logging.info("There are %d critical paths" % len(critical_paths))
        logging.info("Will process these in %d chunks" % n_chunks)
        n_paths_per_chunk = len(critical_paths) // n_chunks
        logging.info("N paths per chunk: %d" % n_paths_per_chunk)

        starting_indexes = list(range(0, len(critical_paths), n_paths_per_chunk))
        ending_indexes = starting_indexes[1:] + [len(critical_paths)]
        chunks = list((s, e) for s, e in zip(starting_indexes, ending_indexes))

        logging.info("Will process chunks (from,to): %s" % chunks)

        t = time.perf_counter()
        flat_kmers = []
        for i, results in enumerate(pool.imap(index_single_thread, zip(itertools.repeat(args_shared), chunks))):
            flat_kmers.append(results)
            logging.info("Done with chunk. Found %d kmers" % (len(results._hashes)))

        logging.info("Time spent to make indexes: %.2f" % (time.perf_counter()-t))
        close_shared_pool()

        flat_kmers = FlatKmers.from_multiple_flat_kmers(flat_kmers)
        logging.info("N kmers in flat kmers: %d" % len(flat_kmers._hashes))
        if args["include_reverse_complement"]:
            logging.info("Adding reverse complements")
            revcomp_flat_kmers = flat_kmers.get_reverse_complement_flat_kmers(args["kmer_size"])
            flat_kmers = FlatKmers.from_multiple_flat_kmers([flat_kmers, revcomp_flat_kmers])
            logging.info("N kmers in flat kmers after revcomp added: %d" % len(flat_kmers._hashes))

        flat_kmers.to_file(args["out_file_name"])
        logging.info("Wrote final FlatKmers to %s" % args["out_file_name"])


    subparser = subparsers.add_parser("index")
    subparser.add_argument("-g", "--graph", required=True, type=Graph.from_file)
    subparser.add_argument("-c", "--critical_graph_paths", required=False, type=from_file, help="Will be created if not specified")
    subparser.add_argument("-p", "--position_id", required=False, type=from_file, help="Will be created if not specified")
    subparser.add_argument("-k", "--kmer-size", type=int, default=31, required=False)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument("-t", "--n-threads", required=False, type=int, default=1)
    #subparser.add_argument("-w", "--whitelist", required=False, type=lambda e: NpSet(FlatKmers.from_file(e)._hashes), help="Only store kmers that are in the whitelist")
    subparser.add_argument("-w", "--whitelist", required=False, type=lambda e: CollisionFreeKmerIndex.from_file(e), help="Only store kmers that are in the whitelist")
    subparser.add_argument("-r", "--include-reverse-complement", required=False, type=bool, default=False)
    subparser.add_argument("-O", "--only-save-one-node-per-kmer", required=False, type=bool, default=False)
    subparser.add_argument("-v", "--max-variant-nodes", required=False, type=int, default=5, help="Max variant nodes allowed in kmer.")
    subparser.set_defaults(func=index)


    def find_critical_paths(args):
        from .critical_graph_paths import CriticalGraphPaths
        graph = Graph.from_file(args.graph)
        critical_paths = CriticalGraphPaths.from_graph(graph, args.kmer_size)
        critical_paths._make_index()
        to_file(critical_paths, args.out_file_name)
        logging.info("Wrote to file %s" % args.out_file_name)

    subparser = subparsers.add_parser("find_critical_paths")
    subparser.add_argument("-g", "--graph", required=True)
    subparser.add_argument("-k", "--kmer-size", type=int, default=31, required=False)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.set_defaults(func=find_critical_paths)


    def add_reverse_complements(args):
        flat = FlatKmers.from_file(args.flat_kmers)
        rev_comp_flat = flat.get_reverse_complement_flat_kmers(k=args.kmer_size)
        flat = FlatKmers.from_multiple_flat_kmers([flat, rev_comp_flat])
        flat.to_file(args.out_file_name)
        logging.info("Saved new flat kmers with reverse complements to %s" % args.out_file_name)

    subparser = subparsers.add_parser("add_reverse_complements")
    subparser.add_argument("-f", "--flat-kmers", required=True)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument("-k", "--kmer-size", type=int, required=True)
    subparser.set_defaults(func=add_reverse_complements)


    def count_kmers(args):
        from .kmer_counter import KmerCounter
        counter = KmerCounter.from_flat_kmersv2(args.flat_kmers, args.modulo, args.subsample_ratio)
        to_file(counter, args.out_file_name)
        logging.info("Wrote counter to %s" % args.out_file_name)

    subparser = subparsers.add_parser("count_kmers")
    subparser.add_argument("-f", "--flat-kmers", required=True, type=FlatKmers.from_file)
    subparser.add_argument("-o", "--out-file-name", required=True)
    subparser.add_argument("-m", "--modulo", required=False, type=int, default=200000033)
    subparser.add_argument("-s", "--subsample-ratio", required=False, type=int, default=1, help="1 to keep every kmer, 2 for every other etc")
    subparser.set_defaults(func=count_kmers)


    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)
    remove_shared_memory_in_session()

