from networkx import DiGraph
from networkx.algorithms import bipartite
from networkx.algorithms.bipartite.generators import configuration_model
from networkx.algorithms.bipartite.generators import gnmk_random_graph
from networkx.utils.random_sequence import powerlaw_sequence
import argparse
import logging
import networkx as nx
import numpy as np
import os
import pickle
import random
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Custom bipartite graph generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('n', type=int, help='# of top nodes')
    parser.add_argument('type', type=str, choices=['classic', 'modified'])
    parser.add_argument('--size', type=int, default=1, help='# of cascades in the Independent Cascade Model')
    args = parser.parse_args()

    n = args.n
    if args.size is None:
        args.size = 1
    graphs = []
    for i in range(args.size):
        if args.type == 'classic':
            logging.basicConfig(level=logging.INFO)
            logging.info('Creating graph #' + str(i + 1))
            B = DiGraph()
            B.add_nodes_from([0, 1, 2], bipartite=0)
            B.add_nodes_from(range(3, 2*n + 2), bipartite=1)
            edge_list0 = [(1, 3)]
            edge_list1 = []
            for j in range(4, n+3):
                edge_list0.append((0, j))
                edge_list0.append((1, j))
            for k in range(n+4, 2*n+3):
                edge_list1.append((2, k))
            print(len(edge_list0))
            print(len(edge_list1))
            edge_list = edge_list0 + edge_list1
            print(edge_list)
            B.add_edges_from(edge_list)

            if nx.is_bipartite(B):
                logging.info("\n ... graph is bipartite.")
            else:
                logging.info("\n ... graph is NOT bipartite.")
            numOfNodes = B.number_of_nodes()
            numOfEdges = B.number_of_edges()
            logging.info('\n...done. Created a bipartite graph with %d nodes and %d edges' % (numOfNodes, numOfEdges))

            evens = {0}
            odds = set(range(1, 2*n+2))
            target_partitions = {0: evens, 1: odds}  # create partitions
        else:
            logging.basicConfig(level=logging.INFO)
            logging.info('Creating graph #' + str(i + 1))
            B = DiGraph()
            B.add_nodes_from([0, 1, 2, 3], bipartite=0)
            B.add_nodes_from(range(4, 3*n + 2), bipartite=1)
            edge_list1 = [(0, 4), (1, 4), (1, 5)]
            edge_list2 = [(2, 5)]
            edge_list3 = [(3, 4)]
            for j in range(6, n+5):
                edge_list1.append((0, j))
                edge_list1.append((1, j))
            for k in range(n+5, 2*n+4):
                edge_list2.append((2, k))
            for l in range(2*n+3, 3*n+2):
                edge_list3.append((3, l))
            print(len(edge_list1))
            print(len(edge_list2))
            print(len(edge_list3))
            edge_list = edge_list1 + edge_list2 + edge_list3
            print(edge_list)
            B.add_edges_from(edge_list)

            if nx.is_bipartite(B):
                logging.info("\n ... graph is bipartite.")
            else:
                logging.info("\n ... graph is NOT bipartite.")
            numOfNodes = B.number_of_nodes()
            numOfEdges = B.number_of_edges()
            logging.info('\n...done. Created a bipartite graph with %d nodes and %d edges' % (numOfNodes, numOfEdges))

            evens = {0, 3}
            odds = set(range(2, 3*n+2))
            target_partitions = {0: evens, 1: odds}  # create partitions
        graphs.append(B)
        i += 1

    # plt.figure()
    # top = nx.bipartite.sets(B)[0]
    # nx.draw(B)
    # plt.savefig('custom_graph.png', bbox_inches="tight")

    with open("datasets/custom" + str(args.size) + "_" + str(n), "w") as f:
        pickle.dump(graphs, f)

    with open("datasets/custom" + str(args.size) + "_" + str(n) + "_partitions", "w") as f:
        pickle.dump(target_partitions, f)

