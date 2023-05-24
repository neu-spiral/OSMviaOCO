import copy
import itertools
import random

import helpers
from helpers import partition_matroid_round, sample_spherical, taverage
from itertools import product
from time import time
from typing import Type, List
import numpy as np
import cvxpy as cp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from operator import concat
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)
from oco_tools import *


def generate_non_stationary_problem():
    np.random.seed(42)
    w = 30
    n_collections = 20

    pts = {}
    for i, pt in enumerate(itertools.product(range(w), range(w))):
        pts[i] = pt
    collections = {}
    collections = []
    for circle in range(n_collections):
        circle_x, circle_y, circle_rad = (np.random.choice(w), np.random.choice(w), np.random.uniform(4, 8))
        collection = []
        for i in pts:
            x, y = pts[i]
            if (in_circle(circle_x, circle_y, circle_rad, x, y)):
                collection.append(i)
        collections.append(collection)
    collections_ids = np.arange(len(collections))
    selection = list(range(len(collections)))
    np.random.shuffle(selection)
    collectionsA = [collections_ids[i] for i in selection[:n_collections // 2]]
    collectionsB = [collections_ids[i] for i in selection[n_collections // 2:]]
    for cols_ids in [collectionsA, collectionsB]:
        for collection_id in cols_ids:
            collection_xy = [pts[pt_i] for pt_i in collections[collection_id]]
            # plt.scatter(np.array(collection_xy)[:, 0], np.array(collection_xy)[:, 1])
        # plt.show()
    # Construct objectives.
    objectives = []
    for cols in [collectionsA, collectionsB]:
        d = 0
        n = len(collections)
        weights = np.ones(len(pts))
        S = []
        b = []
        C = []
        w = []
        c = []
        for i in pts:
            s = list(filter(lambda collection_id: i in collections[collection_id], cols))
            if len(s) > 0:
                if s in S:
                    index = S.index(s)
                    c[index] += weights[i]
                else:
                    S.append(s)
                    w.append(np.ones(n))
                    C.append(len(C))
                    b.append(1)
                    c.append(weights[i])
        objective = ThresholdObjective({'n': n,
                                        'c': c,
                                        'S': S,
                                        'w': w,
                                        'b': b,
                                        'C': C})
        objectives.append(objective)

    avg_obj = ThresholdObjective({'n': n,
                                  'c': list(np.array(objectives[0].c) / 2) + list(np.array(objectives[1].c) / 2),
                                  'S': list(objectives[0].S) + list(objectives[1].S),
                                  'w': list(objectives[0].w) + list(objectives[1].w),
                                  'b': list(objectives[0].b) + list(objectives[1].b),
                                  'C':list(range(len(objectives[0].C)+len(objectives[1].C)))})

    return objectives, avg_obj


import pickle


def dynamic_exp(stationary=False):
    np.random.seed(42)
    objectives, avg_objective = generate_non_stationary_problem()
    n = objectives[0].n
    decision_set = RelaxedPartitionMatroid(n, cardinalities_k=[5],
                                           sets_S=[list(range(20))])

    policyOMA = ShiftedNegativeEntropyOMD(decision_set=decision_set, objective=objectives[0], eta=.001, gamma=.001)

    policyTBG = OnlineTBG(decision_set=decision_set, objective=objectives[0], eta=1, n_colors=1)
    policyFSF = FSF(decision_set=decision_set, objective=objectives[0], eta=.05, gamma=.001)
    T = 50

    def run_non_stationary_exp(policy, T_half=T // 2):
        if stationary:
            policy.objective = avg_objective
        for t in range(T_half):
            policy.step()
        if not stationary:
            policy.objective = objectives[1]
        for t in range(T_half):
            policy.step()
        # plt.plot(taverage(policy.frac_rewards), label=name)
        return (taverage(policy.int_rewards))

    #### OGA ####
    for eta in [.01]:
        rewards = []
        for seed in range(42, 42 + 5):
            np.random.seed(seed)
            policy = OGA(decision_set=decision_set, objective=objectives[0], eta=eta)
            rewards.append(run_non_stationary_exp(policy))
        rewards_mean = np.mean([r for r in rewards], axis=0)
        rewards_std = np.std([r for r in rewards], axis=0)
        lb = rewards_mean - rewards_std
        ub = rewards_mean + rewards_std
        plt.fill_between(np.arange(lb.size) + 1, lb, ub, facecolor='C0', alpha=0.1, color='C0')
        plt.plot(np.arange(lb.size) + 1, rewards_mean, color='C0', label='OGA')

    #### END OGA ####

    ##### OMA ####
    rewards = []
    for seed in range(42, 42 + 5):
        np.random.seed(seed)
        policy = ShiftedNegativeEntropyOMD(decision_set=decision_set, objective=objectives[0], eta=.05, gamma=0)
        rewards.append(run_non_stationary_exp(policy))
    rewards_mean = np.mean([r for r in rewards], axis=0)
    rewards_std = np.std([r for r in rewards], axis=0)
    lb = rewards_mean - rewards_std
    ub = rewards_mean + rewards_std
    plt.fill_between(np.arange(lb.size) + 1, lb, ub, facecolor='C1', alpha=0.5, color='C1')
    plt.plot(np.arange(lb.size) + 1, rewards_mean, color='C1', label='OMA')

    rewards = []
    for seed in range(42, 42 + 5):
        np.random.seed(seed)
        policy = ShiftedNegativeEntropyOMD(decision_set=decision_set, objective=objectives[0], eta=.05, gamma=.02)
        rewards.append(run_non_stationary_exp(policy))
    rewards_mean = np.mean([r for r in rewards], axis=0)
    rewards_std = np.std([r for r in rewards], axis=0)
    lb = rewards_mean - rewards_std
    ub = rewards_mean + rewards_std
    plt.fill_between(np.arange(lb.size) + 1, lb, ub, facecolor='C2', alpha=0.5, color='C2')
    plt.plot(np.arange(lb.size) + 1, rewards_mean, color='C2', label='OMA$^\star$')
    # #### END OMA ###

    # #### FSF ####
    rewards = []
    for seed in range(42, 42 + 5):
        np.random.seed(seed)
        policy = FSF(decision_set=decision_set, objective=objectives[0], eta=.05, gamma=.02)
        rewards.append(run_non_stationary_exp(policy))
    rewards_mean = np.mean([r for r in rewards], axis=0)
    rewards_std = np.std([r for r in rewards], axis=0)
    lb = rewards_mean - rewards_std
    ub = rewards_mean + rewards_std
    plt.fill_between(np.arange(lb.size) + 1, lb, ub, facecolor='C3', alpha=0.5, color='C3')
    plt.plot(np.arange(lb.size) + 1, rewards_mean, color='C3', label='FSF$^\star$')
    #### END FSF ###

    #### TBGD ####
    rewards = []
    for seed in range(42, 42 + 5):
        np.random.seed(seed)
        policy = OnlineTBG(decision_set=decision_set, objective=objectives[0], eta=.05, n_colors=1)
        rewards.append(run_non_stationary_exp(policy))
    rewards_mean = np.mean([r for r in rewards], axis=0)
    rewards_std = np.std([r for r in rewards], axis=0)
    lb = rewards_mean - rewards_std
    ub = rewards_mean + rewards_std
    plt.fill_between(np.arange(lb.size) + 1, lb, ub, facecolor='C4', alpha=0.5, color='C4')
    plt.plot(np.arange(lb.size) + 1, rewards_mean, color='C4', label='TabularGreedy')
    # #### END TBGD ###
    plt.plot([0, T], [avg_objective.get_opt(decision_set)[0]] * 2, color='black', linestyle='--', label='Static $F^\star$')
    if not stationary:
        plt.plot([0, T], [objectives[0].get_opt(decision_set)[0]*.5 + objectives[1].get_opt(decision_set)[0]*.5] * 2, color='black', linestyle='-', label='Dynamic $F^\star$')

    plt.xlim(0,T*1.05)
    plt.ylim(100,(objectives[0].get_opt(decision_set)[0]*.5 + objectives[1].get_opt(decision_set)[0]*.5)*1.05)
    plt.xlabel('Timeslots')
    plt.ylabel(r'$\bar{F}_{\mathcal{X}}$')
    plt.legend(loc='lower right', ncol=2)
    if stationary:
        plt.savefig('results/plots/stationary.pdf')
    else:
        plt.savefig('results/plots/dynamic.pdf')

    plt.show()


def meta_test():
    np.random.seed(42)
    objectives, avg_objective = generate_non_stationary_problem()
    n = objectives[0].n
    decision_set = RelaxedPartitionMatroid(n, cardinalities_k=[5],
                                           sets_S=[list(range(20))])
    T = 50

    def run_non_stationary_exp(policy, T_half=T // 2):
        policy.objective = avg_objective
        for t in range(T_half):
            policy.step()
        for t in range(T_half):
            policy.step()
        return (taverage(policy.int_rewards))

    for eta in [.01]:
        rewards = []
        rewards_meta = {}
        etas = [0.0005, 0.001, .002, .004]
        for eta in etas:
            rewards_meta[eta] = []
        for seed in range(42, 42 + 5):
            np.random.seed(seed)
            policy = MetaPolicy(decision_set, objective=avg_objective, Policy=OGA, etas=etas)

            # fractions = [r'$\frac{1}{1000}$', r'$\frac{1}{200}$', r'$\frac{1}{100}$', r'$\frac{1}{20}$', ]
            rewards.append(run_non_stationary_exp(policy))
            for i, eta in enumerate(etas):
                rewards_meta[eta].append(taverage(policy.experts[i].int_rewards))
        rewards_mean = np.mean([r for r in rewards], axis=0)
        rewards_std = np.std([r for r in rewards], axis=0)
        lb = rewards_mean - rewards_std
        ub = rewards_mean + rewards_std
        plt.fill_between(np.arange(lb.size) + 1, lb, ub, facecolor='C0', alpha=0.25, color='C0')
        plt.plot(np.arange(lb.size) + 1, rewards_mean, color='C0', label='Meta-Policy')
        for eta in rewards_meta:
            rewards = rewards_meta[eta]
            rewards_mean = np.mean([r for r in rewards], axis=0)
            rewards_std = np.std([r for r in rewards], axis=0)
            lb = rewards_mean - rewards_std
            ub = rewards_mean + rewards_std
            plt.fill_between(np.arange(lb.size) + 1, lb, ub, facecolor='C1', alpha=0.25, color='C1')
            if eta ==etas[-1]:
                plt.plot(np.arange(lb.size) + 1, rewards_mean, color='C1', label='OGA ($\eta$)')
            else:
                plt.plot(np.arange(lb.size) + 1, rewards_mean, color='C1')
    plt.xlim(0,T*1.05)
    plt.ylim(100,avg_objective.get_opt(decision_set)[0]*1.05)
    plt.xlabel('Timeslots')
    plt.ylabel(r'$\bar{F}_{\mathcal{X}}$')
    plt.plot([0, T], [avg_objective.get_opt(decision_set)[0]] * 2, color='black', linestyle='--', label='Static $F^\star$')
    plt.legend(loc='lower right', ncol=2)

    plt.savefig('results/plots/meta.pdf')
    plt.show()

def optimism_exp():
    np.random.seed(42)
    objectives, avg_objective = generate_non_stationary_problem()
    n = objectives[0].n
    decision_set = RelaxedPartitionMatroid(n, cardinalities_k=[5],
                                           sets_S=[list(range(20))])
    T = 50

    def run_non_stationary_exp(policy,noise_level= None):
        # policy.objective = avg_objective
        for t in range(T):
            policy.objective =(objectives[t%2])
            if noise_level is not None:
                pred = (objectives[t%2]).supergradient(policy.decision) + np.random.normal(0, size=policy.decision.size) * noise_level
                policy.step(pred=pred)
            else:
                policy.step()
        return (taverage(policy.int_rewards))
    style = ['-', '--', ':']
    for eta_i, eta in enumerate( [.02, 0.001]):
        rewards = []
        for seed in range(42, 42+5):
            np.random.seed(seed)
            policy = OptimisticPolicy(decision_set, objective=avg_objective,  eta=eta, Policy=OGA)
            rewards.append(run_non_stationary_exp(policy, noise_level = 10))
        rewards_mean = np.mean([r for r in rewards], axis=0)
        rewards_std = np.std([r for r in rewards], axis=0)
        lb = rewards_mean - rewards_std
        ub = rewards_mean + rewards_std
        plt.fill_between(np.arange(lb.size) + 1, lb, ub, facecolor='C0', alpha=0.25, color='C0')
        plt.plot(np.arange(lb.size) + 1, rewards_mean, color='C0', label=f'Optimistic OGA ($n_\sigma=10$)' if eta_i ==0 else None, linestyle=style[eta_i])




    for eta_i, eta in enumerate( [.02, 0.001]):
        rewards = []
        for seed in range(42, 42+5):
            np.random.seed(seed)
            policy = OptimisticPolicy(decision_set, objective=avg_objective,  eta=eta, Policy=OGA)
            rewards.append(run_non_stationary_exp(policy, noise_level = 100))
        rewards_mean = np.mean([r for r in rewards], axis=0)
        rewards_std = np.std([r for r in rewards], axis=0)
        lb = rewards_mean - rewards_std
        ub = rewards_mean + rewards_std
        plt.fill_between(np.arange(lb.size) + 1, lb, ub, facecolor='C1', alpha=0.25, color='C1')
        plt.plot(np.arange(lb.size) + 1, rewards_mean, color='C1', label=f'Optimistic OGA ($n_\sigma=100$)' if eta_i ==0 else None, linestyle=style[eta_i])


    for eta_i, eta in enumerate([.02, 0.001]):
        rewards = []
        for seed in range(42, 42 + 5):
            np.random.seed(seed)
            policy = OGA(decision_set, objective=avg_objective, eta=eta)
            rewards.append(run_non_stationary_exp(policy))
        rewards_mean = np.mean([r for r in rewards], axis=0)
        rewards_std = np.std([r for r in rewards], axis=0)
        lb = rewards_mean - rewards_std
        ub = rewards_mean + rewards_std
        plt.fill_between(np.arange(lb.size) + 1, lb, ub, facecolor='C2', alpha=0.25, color='C2')
        plt.plot(np.arange(lb.size) + 1, rewards_mean, color='C2',
                 label=f'OGA' if eta_i ==0 else None, linestyle=style[eta_i])
    plt.xlim(0, T * 1.05)
    # plt.ylim(100, (objectives[0].get_opt(decision_set)[0] * .5 + objectives[1].get_opt(decision_set)[0] * .5)*1.05)
    plt.xlabel('Timeslots')
    plt.ylabel(r'$\bar{F}_{\mathcal{X}}$')
    plt.plot([0, T], [avg_objective.get_opt(decision_set)[0]] * 2, color='black', linestyle='--', label='Static $F^\star$')
    plt.plot([0, T], [objectives[0].get_opt(decision_set)[0] * .5 + objectives[1].get_opt(decision_set)[0] * .5] * 2,
             color='black', linestyle='-', label='Dynamic $F^\star$')
    plt.legend(loc='lower right', ncol=1)
    plt.savefig('results/plots/optimistic.pdf')
    plt.show()

if __name__ == "__main__":


    dynamic_exp(False)
    dynamic_exp(True)
    meta_test()
    optimism_exp()
