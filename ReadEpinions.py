from networkx import DiGraph
from networkx.readwrite.edgelist import read_edgelist, write_edgelist
import logging
import numpy as np
import os
import pickle
import random
import sys


if __name__ == "__main__":

    percent = 1

    logging.basicConfig(level=logging.INFO)
    logging.info('Reading graph...')
    G = read_edgelist("datasets/soc-Epinions1.txt", comments='#', create_using=DiGraph(), nodetype=int)
    numOfNodes = G.number_of_nodes()
    numOfEdges = G.number_of_edges()
    logging.info('\n...done. Read a directed graph with %d nodes and %d edges' % (numOfNodes, numOfEdges))
    degrees = dict(list(G.out_degree(G.nodes())))
    descending_degrees = sorted(degrees.values(), reverse=True)
    evens = set(descending_degrees[::2])
    odds = set(descending_degrees[1::2])
    indices = sorted(range(1, len(degrees.values()) + 1), key=lambda k: degrees.values()[k - 1], reverse=True)
    logging.info('\nTaking a fraction of the graph...')
    n = 100  # (numOfNodes * percent) / 100
    top_n_indices = indices[:5000]
    # G = G.subgraph(top_n_indices).copy()
    rand_n_nodes = random.sample(top_n_indices, n)
    G = G.subgraph(rand_n_nodes).copy()
    numOfNodes = G.number_of_nodes()
    numOfEdges = G.number_of_edges()
    evens = evens.intersection(set(G.nodes()))
    odds = odds.intersection(set(G.nodes()))
    target_partitions = {0: evens, 1: odds}  # create partitions based on their position in the descending_degrees list
    logging.info('\n...done. Created a subgraph with %d nodes and %d edges' % (numOfNodes, numOfEdges))

    logging.info('\nCreating cascades...')
    p = 0.1  # infection probability used in the independent cascade model
    numberOfCascades = 10
    graphs = []
    for cascade in range(numberOfCascades):
        newG = DiGraph()
        newG.add_nodes_from(G.nodes())
        choose = np.array([np.random.uniform(0, 1, G.number_of_edges()) < p, ] * 2).transpose()
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
    logging.info(']n...done. Created %d cascades with %s infection probability.' % (numberOfCascades, p))

    # graphs = []
    # for cascade in os.listdir("edge_lists"):
        # logging.info('Creating cascade %d...' % ())
        # G = read_edgelist("edge_lists/" + cascade, create_using=DiGraph(), nodetype=int)
        # graphs.append(G)

    # G = DiGraph()
    # # G.add_nodes_from([1, 2, 3])
    # # G.add_edges_from([(1, 2), (2, 3)])
    # G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # G.add_edges_from([(1, 2), (2, 3), (2, 8), (5, 7), (6, 8), (2, 10), (3, 4), (4, 5), (4, 6), (6, 3), (10, 9),
    # (10, 3), (7, 8)])
    # # graphs = [new_graph]
    # graphs = [G]
    # graphs = [DiGraph()]
    # graphs[0].add_nodes_from([1, 2, 3])
    # graphs[0].add_edges_from([(1, 2), (2, 3)])
    # graphs.append(DiGraph())
    # graphs[1].add_nodes_from([1, 2, 3])
    # graphs.append(DiGraph())
    # graphs[2].add_nodes_from([1, 2, 3])
    # graphs.append(DiGraph())
    # graphs[3].add_nodes_from([1, 2, 3])
    # graphs.append(DiGraph())
    # graphs[4].add_nodes_from([1, 2, 3])
    # graphs.append(DiGraph())
    # graphs[5].add_nodes_from([1, 2, 3])
    # graphs[5].add_edges_from([(2, 3)])
    # graphs.append(DiGraph())
    # graphs[6].add_nodes_from([1, 2, 3])
    # graphs[6].add_edges_from([(2, 3)])
    # graphs.append(DiGraph())
    # graphs[7].add_nodes_from([1, 2, 3])
    # graphs[7].add_edges_from([(2, 3)])
    # graphs.append(DiGraph())
    # graphs[8].add_nodes_from([1, 2, 3])
    # graphs[8].add_edges_from([(2, 3)])
    # graphs.append(DiGraph())
    # graphs[9].add_nodes_from([1, 2, 3])
    # graphs[9].add_edges_from([(1, 2), (2, 3)])
    # for i in range(10):
    #     sys.stderr.write("edge list of graph #" + str(i) + " : " + str(graphs[i].edges()) + '\n')


    # with open("datasets/random10v2", "w") as f:
    # with open("datasets/epinions_" + str(percent) + "percent_" + str(numberOfCascades) + "cascades", "w") as f:
    with open("datasets/epinions_rand_" + str(n) + "_" + str(numberOfCascades) + "cascades", "w") as f:
        pickle.dump(graphs, f)

    # with open("datasets/epinions_" + str(percent) + "percent_partitions", "w") as f:
    with open("datasets/epinions_rand_" + str(n) + "_partitions", "w") as f:
        pickle.dump(target_partitions, f)

