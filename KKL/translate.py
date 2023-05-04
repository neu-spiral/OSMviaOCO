import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math

from problem import Problem, InfluenceMaximization
from mapping import WDNFMapping
from offline_alg import ApproxGreedy
from online_alg import KKL
from game import Game

class Translator:
    def __init__(self, problem:Problem):
        self.n = problem.problemSize # dimension of the decision/action
        wdnf_dict = problem.wdnf_dict
        self.T = len(wdnf_dict)

        self.wdnfs = [wdnf_dict[t].coefficients for t in range(self.T)]
        self.sign = wdnf_dict[0].sign
        self.translator(self.wdnfs)
        self.find_ws(self.wdnfs, self.set_to_index)

    def translator(self, wdnfs:dict):
        '''
            Translates problem formulated as a list of wdfn functions to a "playing games" problem
        '''
        S = set()
        for wdnf in wdnfs:
            keys = set([tuple(sorted(key))for key in wdnf.keys()])
            S = S.union(keys)

        set_to_index = {}
        index_to_set = {}
        i = 0
        for s in S:
            set_to_index[s] = i
            index_to_set[i] = s
            i += 1
        
        self.set_to_index = set_to_index
        self.index_to_set = index_to_set
        self.m = len(S)
        self.S = S

    def find_w(self, wdnf:dict, set_to_index:set):
        ''' 
            Input: 
                wdnf function coefficients dictionary
                mapping from sets to indices
            Output: w vector
        '''
        w = np.zeros(self.m)
        for key, val in wdnf.items():
            ind = set_to_index[key]
            w[ind] = val
        return w

    def find_ws(self, wdnfs:list[dict], set_to_index:set):
        '''
            Input: 
                wdnfs, list of wdnf functions coefficients dictionary
                set_to_index, mapping from sets to indices
            Output: list of w vectors
        '''
        self.ws = [self.find_w(wdnf, set_to_index) for wdnf in wdnfs]

if __name__ == "__main__":
    graph = nx.gn_graph(10) 
    graphs = [graph] * 10
    constraints = 5
    problem = InfluenceMaximization(graphs, constraints)
    translator = Translator(problem)
    ws = translator.ws
    n = translator.n # dimension of action s
    m = translator.m # dimension of Phi(s)
    T = translator.T # horizon
    sign = translator.sign # sign used in the wdnf functions
    index_to_set = translator.index_to_set
    ground_set = problem.groundSet

    mapping = WDNFMapping(n, index_to_set, sign)
    linear_solver = problem.get_solver()
    initial_point = problem.get_initial_point()
    approx_alg = ApproxGreedy(linear_solver, mapping, initial_point, n)


    # set constants
    W = max([np.linalg.norm(w) for w in ws]) # ||w|| <= W
    R = np.sqrt(m) # ||Phi(s)|| <= R
    alpha = math.e
    delta = (alpha + 1) * R**2 / T
    eta = (alpha + 1) * R / (W * np.sqrt(T))

    # initialize online algorithm
    alg = KKL(approx_alg, mapping, alpha, delta, eta, R, n)

    # initialize game
    game = Game(alg, mapping, ws, n, T)

    game.play()

    timesteps = np.arange(T) + 1
    cum_avg_reward = game.get_cum_avg_reward()
    
    
    plt.plot(timesteps, cum_avg_reward, "-o")
    plt.xlabel("timestep")
    plt.ylabel("average reward")
    plt.title("Cummulative average reward")
    plt.show()