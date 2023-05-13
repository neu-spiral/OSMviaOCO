import argparse
import logging
import numpy as np
import pickle
from random import choices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Cascades from ZKC data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n', type=int, default=100, help='# of individuals')
    parser.add_argument('--topics', type=int, default=10, help='# of topics')
    parser.add_argument('--T', type=int, default=100, help='time horizon')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info('Generating a team formation dataset...')

    seed = args.seed
    T = args.T  
    num_of_topics = args.topics
    n = args.n 

    np.random.seed(seed = seed)
    max_num_of_citations = 100
    min_num_of_citations = 0
    mu = 60
    sigma = 10

    # Contruct 2 partitions
    evens = set(range(0, n, 2))
    odds = set(range(1, n, 2))
    partitions = {0: evens, 1: odds}

    topic_funcs = []
    for t in range(num_of_topics):
        citations = np.random.normal(mu, sigma, n)
        citations = np.minimum([max_num_of_citations] * n, citations)
        citations = np.maximum([min_num_of_citations] * n, citations)
        h = citations
        H = np.zeros(n**2).reshape(n, n)
        for i in range(n-1):
            for j in range(i+1, n):
                H[i][j] = H[j][i] = np.min(np.random.normal(-50, 10, 1), 0)
        ones = np.ones(n)
        while True:
            vec = h + np.dot(H, ones)
            if np.all(vec >= 0):
                break
            H = H / 1.01
        topic_funcs.append((h.tolist(), H.tolist()))

    funcs = choices(topic_funcs, k=T)

    print(f"h_0 = {funcs[0][0]}")
    print(f"H_0 = {funcs[0][1]}")
    print(f"Generated T = {len(funcs)} quadratic functions")

    with open(f"datasets/TF_{n}_{T}_{num_of_topics}_{seed}_partitions", 'wb') as f:
        pickle.dump(partitions, f)
    with open(f"datasets/TF_{n}_{T}_{num_of_topics}_{seed}", 'wb') as f:
        pickle.dump(funcs, f)
