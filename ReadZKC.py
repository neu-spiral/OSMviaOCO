from networkx.readwrite.edgelist import read_edgelist, write_edgelist
import argparse
import logging
import networkx as nx
import numpy as np
import os
import pickle
import random
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Cascades from ZKC data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n', type=int, default=10, help='# of cascades')
    parser.add_argument('--p', type=float, default=0.1, help='probability of infection')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info('Reading the Zachary Karate Club graph...')
    # Network topology
    G = nx.karate_club_graph()
    numOfNodes = G.number_of_nodes()
    numOfEdges = G.number_of_edges()
    logging.info('\n...done. Created a graph with %d nodes and %d edges' % (numOfNodes, numOfEdges))
    degrees = dict(list(G.degree(G.nodes())))
    descending_degrees = sorted(degrees.values(), reverse=True)
    evens = set(descending_degrees[::2])
    odds = set(descending_degrees[1::2])
    target_partitions = {0: evens, 1: odds}  # create partitions based on their position in the descending_degrees list

    logging.info('\nCreating cascades...')
    p = args.p  # infection probability used in the independent cascade model
    numberOfCascades = args.n
    graphs = []
    for cascade in range(numberOfCascades):
        newG = nx.DiGraph()
        newG.add_nodes_from(G.nodes())
        choose = np.array([np.random.uniform(0, 1, G.number_of_edges()) < p, ] * 2).transpose()
        print(len(choose))
        chosen_edges = list(np.extract(choose, G.edges()))
        print(len(chosen_edges))
        chosen_edges = list(zip(chosen_edges[0::2], chosen_edges[1::2]))
        print(len(chosen_edges))
        newG.add_edges_from(chosen_edges)
        graphs.append(newG)
        numOfNodes = graphs[cascade].number_of_nodes()
        numOfInfEdges = graphs[cascade].number_of_edges()
        logging.info('\nCreated cascade #%d with %d nodes and %d edges.' % (cascade, numOfNodes, numOfInfEdges))
    logging.info(']n...done. Created %d cascades with %s infection probability.' % (numberOfCascades, p))

    with open("datasets/ZKC_" + str(numberOfCascades) + "_" + str(p).replace('.', ''), "wb") as f:
        pickle.dump(graphs, f)

    with open("datasets/ZKC_" + str(numberOfCascades) + "_" + str(p).replace('.', '') + "_partitions", "wb") as f:
        pickle.dump(target_partitions, f)
