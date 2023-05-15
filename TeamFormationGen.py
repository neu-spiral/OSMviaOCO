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
    parser.add_argument('--constraint', type=str, default='partitions', help='constraint (partitions or uniform)')
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

    # Constraints
    if args.constraint == 'partitions':
        print(f'partition matroid constraint')
        evens = set(range(0, n, 2))
        odds = set(range(1, n, 2))
        partitions = {0: evens, 1: odds}
    else:
        print(f'uniform matroid constraint')
        partitions = {0: set(range(n))}

    topic_funcs = deque()
    for t in range(num_of_topics):
        print(f'Creating function for topic #{t}')
        citations = np.random.normal(mu, sigma, n)
        citations = np.minimum([max_num_of_citations] * n, citations)
        citations = np.maximum([min_num_of_citations] * n, citations)
        h = citations
        h[0] = h[1] = h[2] = h[3] = 100

        H = np.zeros(n**2).reshape(n, n)
        for i in range(n-1):
            for j in range(i+1, n):
                H[i][j] = H[j][i] = np.min(np.random.normal(-20, 10, 1), 0)
        H[0] = H[1] = H[2] = H[3] = np.zeros(n)
        H[:,0] = H[:,1] = H[:,2] = H[:,3] = np.zeros(n)

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
        topic_funcs.append((h.tolist(), H.tolist()))
    
    funcs = choices(list(topic_funcs), k=T)

    print(f"h_0 = {funcs[0][0]}")
    print(f"H_0 = {funcs[0][1]}")
    print(f"Generated T = {len(funcs)} quadratic functions")   


    with open(f"datasets/TF_{n}_{T}_{num_of_topics}_{args.constraint}", 'wb') as f:
        pickle.dump(partitions, f)
    with open(f"datasets/TF_{n}_{T}_{num_of_topics}", 'wb') as f:
        pickle.dump(funcs, f)
