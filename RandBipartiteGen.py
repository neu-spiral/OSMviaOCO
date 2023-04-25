from networkx import Graph
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random bipartite graph generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('n', type=int, help='# of top nodes')
    parser.add_argument('type', type=str, choices=['uniform', 'powerlaw'], help='Type of the degree distribution')
    parser.add_argument('--m', type=int, help='# of bottom nodes')
    parser.add_argument('--k', type=int, help='# of edges')
    parser.add_argument('--size', type=int, default=1, help='# of cascades in the Independent Cascade Model')
    parser.add_argument('--partitions', type=int, default=1, help='# of partitions')
    args = parser.parse_args()

    n = args.n
    if args.m is not None:
        m = args.m
    else:
        m = n
    if args.k is not None:
        k = args.k
    else:
        k = 4*n
    graphs = []
    for i in range(args.size):
        logging.basicConfig(level=logging.INFO)
        logging.info('Creating graph #' + str(i + 1))
        if args.type == 'uniform':
            B = gnmk_random_graph(n, m, k)
        else:
            a_deg_seq = list(map(int, powerlaw_sequence(n)))
            a_deg_seq.sort(reverse=True)
            k = sum(a_deg_seq)
            b_deg_seq = a_deg_seq
            random.shuffle(b_deg_seq)
            B = configuration_model(a_deg_seq, b_deg_seq, create_using=Graph())
        sys.stderr.write("\nedges of the initial graph are: " + str(B.edges()))
        B = B.to_directed()
        sys.stderr.write("\nedges after to_directed() are: " + str(B.edges()))
        M = {x for x, d in B.nodes(data=True) if d['bipartite'] == 1}
        del_edges = []
        for (u, v) in B.edges():
            if u in M:
                del_edges.append((u, v))
        B.remove_edges_from(del_edges)
        sys.stderr.write("\nedges after deletion are: " + str(B.edges()))
        if nx.is_bipartite(B):
            logging.info("\n ... graph is bipartite.")
        else:
            logging.info("\n ... graph is NOT bipartite.")
        numOfNodes = B.number_of_nodes()
        numOfEdges = B.number_of_edges()
        k = numOfEdges
        logging.info('\n...done. Created a bipartite graph with %d nodes and %d edges' % (numOfNodes, numOfEdges))

        degN, degM = bipartite.degrees(B, M)
        degrees = dict(degN)
        print("degree dict: " + str(degrees))
        descending_degrees = sorted(degrees.values(), reverse=True)
        print("degrees descending: " + str(descending_degrees))
        sorted_nodes = sorted(range(len(degrees.values())), key=lambda c: list(degrees.values())[c], reverse=True)
        print("nodes sorted by descending degrees: " + str(sorted_nodes))
        max_degree = descending_degrees[0]
        min_degree = descending_degrees[-1]
        mean = (sum(degrees.values()) * 1.0) / len(degrees)
        std_dev = np.sqrt((sum([(degrees[i] - mean)**2 for i in degrees]) * 1.0) / len(degrees))
        logging.info('\nMax degree: %s, min degree: %s, mean: %s, and standard deviation: %s.' % (max_degree,
                                                                                                  min_degree, mean,
                                                                                                  std_dev))
        graphs.append(B)
        i += 1

    target_partitions = dict()
    for j in range(args.partitions + 1):
        target_partitions[j] = set(sorted_nodes[j::args.partitions])

    # zeros = set(sorted_nodes[::10])
    # ones = set(sorted_nodes[1::10])
    # twos = set(sorted_nodes[2::10])
    # threes = set(sorted_nodes[3::10])
    # fours = set(sorted_nodes[4::10])
    # fives = set(sorted_nodes[5::10])
    # sixes = set(sorted_nodes[6::10])
    # sevens = set(sorted_nodes[7::10])
    # eights = set(sorted_nodes[8::10])
    # nines = set(sorted_nodes[9::10])
    # target_partitions = {0: zeros, 1: ones, 2: twos, 3: threes, 4: fours, 5: fives, 6: sixes, 7: sevens, 8: eights,
    #                      9: nines}  # create partitions based on their position in the descending_degrees list
    print("target partitions are: " + str(target_partitions))

    with open("datasets/RB" + str(args.size) + args.type + "_" + str(n) + "_" + str(m) + "_" + str(k), "wb") as f:
        pickle.dump(graphs, f)

    with open("datasets/RB" + str(args.size) + args.type + "_" + str(n) + "_" + str(m) + "_" + str(k) + "_partitions",
              "wb") as f:
        pickle.dump(target_partitions, f)

