import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math

import sys

from typing import List

sys.path.append('../')

from ProblemInstances import Problem


class Translator:
    def __init__(self, problem: Problem):
        self.n = problem.problemSize  # dimension of the decision/action
        wdnf_dict = problem.wdnf_dict
        self.T = len(wdnf_dict)

        self.wdnfs = [wdnf_dict[t].coefficients for t in range(self.T)]
        for coeffs in self.wdnfs:  # add a constant term if not already present
            coeffs[()] = 1

        self.sign = wdnf_dict[0].sign
        self.translator(self.wdnfs)
        self.find_ws(self.wdnfs, self.set_to_index)

    def translator(self, wdnfs: dict):
        '''
            Translates problem formulated as a list of wdfn functions to a "playing games" problem
        '''
        S = set()
        for wdnf in wdnfs:
            keys = set([tuple(sorted(key)) for key in wdnf.keys()])
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

    def find_w(self, wdnf: dict, set_to_index: set):
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

    def find_ws(self, wdnfs: List[dict], set_to_index: set):
        '''
            Input: 
                wdnfs, list of wdnf functions coefficients dictionary
                set_to_index, mapping from sets to indices
            Output: list of w vectors
        '''
        self.ws = [self.find_w(wdnf, set_to_index) for wdnf in wdnfs]
