from helpers import save
from io import open
from networkx import Graph
from networkx.algorithms import bipartite
import argparse
import logging
import networkx as nx
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a subset from MovieLens data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n', type=int, default=300, help='# of cascades')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info('Reading the MovieLens graph...')
    file_path = 'datasets/ml-datasets/ml-10M100K/ratings.dat'
    partition_path = 'datasets/ml-datasets/ml-10M100K/movies.dat'
    n = args.n  # subgraph size
    logging.basicConfig(level=logging.INFO)
    logging.info('Reading ratings...')
    with open(file_path, 'r') as f:
        lines = f.readlines()
        users = []
        movies = []
        ratings = []
        for line in lines:
            line = line.split('::')
            users.append(int(line[0]))
            movies.append(line[1])
            ratings.append((int(line[0]), line[1], float(line[2]) / 5.0))
    logging.info('...done.')

    logging.info('Reading movie genres...')
    with open(partition_path, 'r') as f:
        lines = f.readlines()
        target_partitions = dict()
        for line in lines:
            line = line.split('::')
            movie = int(line[0])
            genre = line[-1].split('|')[0].strip()
            if genre in target_partitions:
                target_partitions[genre].add(movie)
                # print(target_partitions[genre])
            else:
                target_partitions[genre] = {movie}
                # print(target_partitions[genre])
    logging.info('...done.')

    B = Graph()
    B.add_nodes_from(movies, bipartite=0)
    B.add_nodes_from(users, bipartite=1)
    B.add_weighted_edges_from(ratings)

    user_degrees = dict(list(B.degree([node for node, d in B.nodes(data=True) if d['bipartite'] == 1])))
    # descending_degrees = sorted(user_degrees.values(), reverse=True)
    user_indices = sorted(range(1, len(user_degrees.values()) + 1), key=lambda k: list(user_degrees.values())[k - 1],
                          reverse=True)
    top_n_user_indices = user_indices[:n]
    movies_sub = list(B.neighbors(top_n_user_indices[0]))[:n]
    print(f"{top_n_user_indices[0]}")
    print(f"movies_sub before {movies_sub}")

    B = B.subgraph(top_n_user_indices + movies_sub).copy()
    # print(set(map(int, movies_sub)))
    B = nx.convert_node_labels_to_integers(B, first_label=0, ordering='default')

    movies = [n for n, d in B.nodes(data=True) if d['bipartite'] == 0]
    users = [n for n, d in B.nodes(data=True) if d['bipartite'] == 1]

    movies_mapping = dict(zip(movies, range(len(movies))))
    users_mapping = dict(zip(users, range(len(movies), len(movies) + len(users))))

    movies_mapping.update(users_mapping)
    nodes_mapping = movies_mapping

    print(f"nodes will be mapped according to {nodes_mapping}")
    B = nx.relabel_nodes(B, nodes_mapping)
    movies = [n for n, d in B.nodes(data=True) if d['bipartite'] == 0]
    users = [n for n, d in B.nodes(data=True) if d['bipartite'] == 1]

    print(f"\n users are: {users}")
    print(f"\n movies are: {movies}")
    print(f"\n ratings are: {list(B.edges(data=True))}")

    for genre in target_partitions:
        target_partitions[genre] = set(map(int, movies)).intersection(target_partitions[genre])
    target_partitions = {k: v for k, v in target_partitions.items() if v != set()}
    for key in target_partitions:
        print('\n' + str(key) + ': ' + str(target_partitions[key]))

    if nx.is_bipartite(B):
        logging.info("\n ... graph is bipartite.")
    else:
        logging.info("\n ... graph is NOT bipartite.")
    numOfNodes = B.number_of_nodes()
    numOfEdges = B.number_of_edges()
    logging.info('\nCreated a graph with %d nodes and %d edges' % (numOfNodes, numOfEdges))

    with open(f"datasets/MovieLens_{len(users)}_users_{len(movies)}_movies", "wb") as f:
        pickle.dump(B, f)

    with open(f"datasets/MovieLens_{len(users)}_users_{len(movies)}_movies_partitions", "wb") as f:
        pickle.dump(target_partitions, f)
