from helpers import save
from networkx import Graph
from networkx.algorithms import bipartite
import logging
import networkx as nx


if __name__ == "__main__":
    file_path = 'datasets/ratings.dat'
    partition_path = 'datasets/movies.dat'
    n = 300  # subgraph size

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
            ratings.append((int(line[0]), line[1], int(line[2]) / 5.0))
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
    descending_degrees = sorted(user_degrees.values(), reverse=True)
    user_indices = sorted(range(1, len(user_degrees.values()) + 1), key=lambda k: user_degrees.values()[k - 1],
                          reverse=True)
    top_n_user_indices = user_indices[:n]
    movies_sub = list(B.neighbors(top_n_user_indices[0]))[:n]

    B = B.subgraph(top_n_user_indices + movies_sub).copy()
    # print(set(map(int, movies_sub)))
    for genre in target_partitions:
        target_partitions[genre] = set(map(int, movies_sub)).intersection(target_partitions[genre])
    for key in target_partitions:
        print('\n' + str(key) + ': ' + str(target_partitions[key]))

    print('\n users are:' + str([node for node, d in B.nodes(data=True) if d['bipartite'] == 1]))
    print('\n movies are:' + str([node for node, d in B.nodes(data=True) if d['bipartite'] == 0]))
    print('\n ratings are: ' + str(list(B.edges(data=True))))

    if nx.is_bipartite(B):
        logging.info("\n ... graph is bipartite.")
    else:
        logging.info("\n ... graph is NOT bipartite.")
    numOfNodes = B.number_of_nodes()
    numOfEdges = B.number_of_edges()
    logging.info('\nCreated a graph with %d nodes and %d edges' % (numOfNodes, numOfEdges))

    save(file_path.replace('.dat', '') + '_' + str(n), B)
    save(partition_path.replace('.dat', '') + '_' + str(n) + '_partitions', target_partitions)

    # save(file_path.replace('.dat', ''), B)
    # save(partition_path.replace('.dat', '') + '_partitions', target_partitions)
