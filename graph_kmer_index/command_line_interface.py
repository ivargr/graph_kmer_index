import sys
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
from offsetbasedgraph import Graph, SequenceGraph, NumpyIndexedInterval
from vcfmap import VcfMap
from .index_creator import IndexCreator
from .snp_kmer_finder import SnpKmerFinder
import pickle
from .flat_kmers import FlatKmers
from .kmer_index import KmerIndex

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
    finder = SnpKmerFinder(graph, sequence_graph, linear_path, k=args.kmer_size)
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
    subparser.add_argument("-k", "--kmer_size", required=False, type=int, default=32)
    subparser.set_defaults(func=create_index)

    subparser = subparsers.add_parser("merge_indexes")
    subparser.add_argument("-o", "--out_file_name", required=True)
    subparser.add_argument("files", nargs="+")
    subparser.set_defaults(func=merge_indexes)



    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    args.func(args)

