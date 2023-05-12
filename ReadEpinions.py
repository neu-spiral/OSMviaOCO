from networkx import DiGraph
from networkx.readwrite.edgelist import read_edgelist, write_edgelist
import logging
import networkx as nx
import numpy as np
import os
import pickle
import random
import sys


if __name__ == "__main__":

    percent = 1
    # list of seeds with len = numOfCascades

    logging.basicConfig(level=logging.INFO)
    logging.info('Reading graph...')
    G = read_edgelist("datasets/soc-Epinions1.txt", comments='#', create_using=DiGraph(), nodetype=int)
    numOfNodes = G.number_of_nodes()
    numOfEdges = G.number_of_edges()
    logging.info('\n...done. Read a directed graph with %d nodes and %d edges' % (numOfNodes, numOfEdges))
    degrees = dict(list(G.out_degree(G.nodes())))
    indices = sorted(range(1, len(degrees.values()) + 1), key=lambda k: list(degrees.values())[k - 1], reverse=True)
    logging.info('\nTaking a fraction of the graph...')
    n = 5000  # (numOfNodes * percent) / 100
    top_n_indices = indices[:n]
    G = G.subgraph(top_n_indices).copy()
    # rand_n_nodes = random.sample(top_n_indices, n)
    # G = G.subgraph(rand_n_nodes).copy()
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    # print(G.nodes())
    numOfNodes = G.number_of_nodes()
    numOfEdges = G.number_of_edges()
    degrees = dict(list(G.out_degree(G.nodes())))
    descending_degrees = sorted(degrees.values(), reverse=True)
    evens = set(descending_degrees[::2])
    odds = set(descending_degrees[1::2])
    target_partitions = {0: evens, 1: odds}  # create partitions based on their position in the descending_degrees list
    logging.info('\n...done. Created a subgraph with %d nodes and %d edges' % (numOfNodes, numOfEdges))

    logging.info('\nCreating cascades...')
    p = 0.1  # infection probability used in the independent cascade model
    numberOfCascades = 500
    graphs = []
    for cascade in range(numberOfCascades):
        newG = DiGraph()
        newG.add_nodes_from(G.nodes())
        choose = np.array([np.random.uniform(0, 1, G.number_of_edges()) < p, ] * 2).transpose()
        # print(len(choose))
        chosen_edges = np.extract(choose, G.edges())
        # print(len(chosen_edges))
        chosen_edges = list(zip(chosen_edges[0::2], chosen_edges[1::2]))
        # print(len(chosen_edges))
        newG.add_edges_from(chosen_edges)
        graphs.append(newG)
        numOfNodes = graphs[cascade].number_of_nodes()
        numOfInfEdges = graphs[cascade].number_of_edges()
        logging.info('\nCreated cascade %d with %d nodes and %d edges.' % (cascade, numOfNodes, numOfInfEdges))
    logging.info(']n...done. Created %d cascades with %s infection probability.' % (numberOfCascades, p))

    with open(f"datasets/epinions_{str(G.nodes())}_nodes_{str(numberOfCascades)}_cascades_p_{str(p).replace('.', 'p')}",
              "wb") as f:
        pickle.dump(graphs, f)

    with open(f"datasets/epinions_{str(G.nodes())}_nodes_{str(numberOfCascades)}_cascades_p_{str(p).replace('.', 'p')}"
              f"_partitions", "wb") as f:
        pickle.dump(target_partitions, f)

