import sys
import argparse
import logging

from .collision_free_kmer_index import CollisionFreeKmerIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
from offsetbasedgraph import Graph, SequenceGraph, NumpyIndexedInterval
from vcfmap import VcfMap
from .index_creator import IndexCreator
from .snp_kmer_finder import SnpKmerFinder
import pickle
from .flat_kmers import FlatKmers
from .kmer_index import KmerIndex
from .reverse_kmer_index import ReverseKmerIndex
from .unique_kmer_index import UniqueKmerIndex


def main():
    run_argument_parser(sys.argv[1:])


def create_index(args):
    logging.info("Loading data")
    graph = Graph.from_file(args.graph_file_name)
    sequence_graph = SequenceGraph.from_file(args.graph_file_name + ".sequences")
    #vcfmap = VcfMap.from_file(args.vcfmap_file_name)
    linear_path = NumpyIndexedInterval.from_file(args.linear_path_file_name)
    #critical_nodes = pickle.load(open(args.critical_nodes, "rb"))
    logging.info("Running kmerfinder")
    #finder = KmerFinder(graph, sequence_graph, critical_nodes, linear_path, k=15, store_all_kmers=False)
    finder = SnpKmerFinder(graph, sequence_graph, linear_path, k=args.kmer_size, spacing=args.spacing,
                           include_reverse_complements=args.include_reverse_complement)
    kmers = finder.find_kmers()
    kmers.to_file(args.out_file_name)
    #creator.to_file(args.out_file_name)


def merge_indexes(args):
    indexes = []
    for file_name in args.files:
        indexes.append(FlatKmers.from_file(file_name))

    index = KmerIndex.from_multiple_flat_kmers(indexes)
    index.to_file(args.out_file_name)
    logging.info("Done")

def make_collision_free(args):
    flat = FlatKmers.from_file(args.flat_index)
    index = CollisionFreeKmerIndex.from_flat_kmers(flat, modulo=args.hash_modulo)
    index.to_file(args.out_file_name)
    logging.info("Done making collision free index")


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
    subparser.add_argument("-l", "--linear_path_file_name", required=True)
    subparser.add_argument("-k", "--kmer_size", required=False, type=int, default=31)
    subparser.add_argument("-r", "--include_reverse_complement", required=False, type=bool, default=False)
    subparser.add_argument("-spacing", "--spacing", required=False, type=int, default=31)
    subparser.set_defaults(func=create_index)

    subparser = subparsers.add_parser("merge_indexes")
    subparser.add_argument("-o", "--out_file_name", required=True)
    subparser.add_argument("files", nargs="+")
    subparser.set_defaults(func=merge_indexes)

    subparser = subparsers.add_parser("make_collision_free")
    subparser.add_argument("-o", "--out_file_name", required=True)
    subparser.add_argument("-f", "--flat-index", required=True)
    subparser.add_argument("-m", "--hash_modulo", required=False, type=int, default=452930477)
    subparser.set_defaults(func=make_collision_free)

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

