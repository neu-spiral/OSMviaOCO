from networkx import DiGraph
from networkx.generators.random_graphs import erdos_renyi_graph
from networkx.readwrite.edgelist import read_edgelist, write_edgelist
import logging
import numpy as np
import os
import pickle
import random
import sys


if __name__ == "__main__":

    n = 100
    p = 0.05
    logging.basicConfig(level=logging.INFO)
    logging.info('Creating graph...')
    G = erdos_renyi_graph(n, p, directed=True)
    numOfNodes = G.number_of_nodes()
    numOfEdges = G.number_of_edges()
    logging.info('\n...done. Created a directed Erdos-Renyi graph with %d nodes and %d edges' % (numOfNodes, numOfEdges))
    degrees = dict(list(G.out_degree(G.nodes())))
    descending_degrees = sorted(degrees.values(), reverse=True)
    evens = set(descending_degrees[::2])
    odds = set(descending_degrees[1::2])
    # evens = evens.intersection(set(G.nodes()))
    # odds = odds.intersection(set(G.nodes()))
    target_partitions = {0: evens, 1: odds}  # create partitions based on their position in the descending_degrees list

    logging.info('\nCreating cascades...')
    p1 = 0.1  # infection probability used in the independent cascade model
    numberOfCascades = 10
    graphs = []
    for cascade in range(numberOfCascades):
        newG = DiGraph()
        newG.add_nodes_from(G.nodes())
        choose = np.array([np.random.uniform(0, 1, G.number_of_edges()) < p1, ] * 2).transpose()
        print(len(choose))
        chosen_edges = np.extract(choose, G.edges())
        print(len(chosen_edges))
        chosen_edges = zip(chosen_edges[0::2], chosen_edges[1::2])
        print(len(chosen_edges))
        newG.add_edges_from(chosen_edges)
        graphs.append(newG)
        numOfNodes = graphs[cascade].number_of_nodes()
        numOfInfEdges = graphs[cascade].number_of_edges()
        logging.info('\nCreated cascade %d with %d nodes and %d edges.' % (cascade, numOfNodes, numOfInfEdges))
    logging.info(']n...done. Created %d cascades with %s infection probability.' % (numberOfCascades, p1))

    with open("datasets/ER_" + str(p).replace('.', '') + '_' + str(n) + "_" + str(numberOfCascades) + "cascades", "w") as f:
        pickle.dump(graphs, f)

    with open("datasets/ER_" + str(p).replace('.', '') + '_' + str(n) + "_partitions", "w") as f:
        pickle.dump(target_partitions, f)

