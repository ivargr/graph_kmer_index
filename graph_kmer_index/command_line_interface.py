import sys
import argparse
import logging
from multiprocessing import shared_memory

from .collision_free_kmer_index import CollisionFreeKmerIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
from obgraph import Graph
from .index_creator import IndexCreator
from .snp_kmer_finder import SnpKmerFinder
import pickle
from .flat_kmers import FlatKmers
from .reverse_kmer_index import ReverseKmerIndex
from .unique_kmer_index import UniqueKmerIndex


def main():
    run_argument_parser(sys.argv[1:])


def create_index(args):
    logging.info("Loading data")
    graph = Graph.from_file(args.graph_file_name)
    logging.info("Running kmerfinder")
    finder = SnpKmerFinder(graph, k=args.kmer_size, spacing=args.spacing,
                           include_reverse_complements=args.include_reverse_complement,
                           pruning=args.pruning,
                           max_kmers_same_position=args.max_kmers_same_position,
                           max_frequency=args.max_frequency,
                           max_variant_nodes=args.max_variant_nodes)
    kmers = finder.find_kmers()
    kmers.to_file(args.out_file_name)
    #creator.to_file(args.out_file_name)


def make_from_flat(args):
    flat = FlatKmers.from_file(args.flat_index)
    index = CollisionFreeKmerIndex.from_flat_kmers(flat, modulo=args.hash_modulo)
    index.to_file(args.out_file_name)
    logging.info("Done making kmer index")


def make_reverse(args):
    flat = FlatKmers.from_file(args.flat_index)
    reverse = ReverseKmerIndex.from_flat_kmers(flat)
    reverse.to_file(args.out_file_name)
    logging.info("Done. Wrote reverse index to file: %s" % args.out_file_name)


def make_unique_index(args):
    graph = Graph.from_file(args.graph)
    reverse = ReverseKmerIndex.from_file(args.reverse)
    flat = FlatKmers.from_file(args.flat_index)
    unique = UniqueKmerIndex.from_flat_kmers_and_snps_graph(flat, graph, reverse)
    unique.to_file(args.out_file_name)


def run_argument_parser(args):
    parser = argparse.ArgumentParser(
        description='Graph Kmer Index.',
        prog='graph_kmer_index',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=100))

    subparsers = parser.add_subparsers()
    subparser = subparsers.add_parser("make")
    subparser.add_argument("-g", "--graph_file_name", required=True)
    subparser.add_argument("-o", "--out_file_name", required=True)
    subparser.add_argument("-k", "--kmer_size", required=False, type=int, default=31)
    subparser.add_argument("-r", "--include_reverse_complement", required=False, type=bool, default=False)
    subparser.add_argument("-s", "--spacing", required=False, type=int, default=31)
    subparser.add_argument("-p", "--pruning", required=False, type=bool, default=False, help="Set to True to prune unecessary kmers")
    subparser.add_argument("-m", "--max-kmers-same-position", required=False, type=int, default=100000, help="Maximum number of kmers allowd to be added from the same ref position")
    subparser.add_argument("-M", "--max-frequency", required=False, type=int, default=100000, help="Skip kmers with frequency higher than this. Will never skip kmers crossing variants.")
    subparser.add_argument("-v", "--max-variant-nodes", required=False, type=int, default=100000, help="Max variant nodes allowed in kmer.")

    subparser.set_defaults(func=create_index)

    subparser = subparsers.add_parser("make_from_flat")
    subparser.add_argument("-o", "--out_file_name", required=True)
    subparser.add_argument("-f", "--flat-index", required=True)
    subparser.add_argument("-m", "--hash_modulo", required=False, type=int, default=452930477)
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

    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)

