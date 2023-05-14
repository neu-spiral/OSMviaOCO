import argparse
import logging
import numpy as np
import pickle
from random import choices
from oco_tools import ThresholdObjective
from collections import deque

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Cascades from ZKC data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n', type=int, default=100, help='# of individuals')
    parser.add_argument('--topics', type=int, default=5, help='# of topics')
    parser.add_argument('--T', type=int, default=100, help='time horizon')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--trace', type=str, default='constant', help='trace')
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
    sigma = 20

    # Contruct 2 partitions
    evens = set(range(0, n, 2))
    odds = set(range(1, n, 2))
    partitions = {0: evens, 1: odds}

    topic_funcs = deque()
    for t in range(num_of_topics):
        print(f'Creating function for topic #{t}')
        citations = np.random.normal(mu, sigma, n)
        citations = np.minimum([max_num_of_citations] * n, citations)
        citations = np.maximum([min_num_of_citations] * n, citations)
        h = citations
        H = np.zeros(n**2).reshape(n, n)
        for i in range(n-1):
            for j in range(i+1, n):
                H[i][j] = H[j][i] = np.min(np.random.normal(-20, 10, 1), 0)
        print(f'Creating function for topic #{t}')

        # shrink rows and cols of H until we achieve submodularity
        ones = np.ones(n)
        while True:
            vec = h + np.dot(H, ones)
            for i in range(len(vec)):
                if h[i] == 0:
                    H[i] = list(np.zeros(n))
                    H[:,i] = list(np.zeros(n))

                if vec[i] < 0:
                    H[i] = H[i] / 1.1 # shrink row
                    H[:,i] = H[:,i] / 1.1 # shrink col
            if np.all(vec >= 0):
                break
        print(f'test')
        topic_funcs.append((h.tolist(), H.tolist()))
    
    funcs = choices(list(topic_funcs), k=T)

    print(f"h_0 = {funcs[0][0]}")
    print(f"H_0 = {funcs[0][1]}")
    print(f"Generated T = {len(funcs)} quadratic functions")    

    with open(f"datasets/TF_{n}_{T}_{num_of_topics}_partitions", 'wb') as f:
        pickle.dump(partitions, f)
    with open(f"datasets/TF_{n}_{T}_{num_of_topics}", 'wb') as f:
        pickle.dump(funcs, f)
