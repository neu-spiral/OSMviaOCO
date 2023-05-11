from abc import ABC, ABCMeta, abstractmethod
from helpers import partition_matroid_round, sample_spherical
from itertools import product
from time import time
from typing import Type
import numpy as np
import cvxpy as cp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class ThresholdObjective:
    def __init__(self, params):
        """Params format:
        params = { The naming corresponds to the one used in the paper
            'n': n, # dimension size - n, |V|
            'C': C, # range list of threshold functions - # of wdnfs C should be a list
            'b': b, # Vector of thresholds - |1|_C
            'w': w, # List of vectors (variable size) w[i][j] is the weight of the x[j] within threshold function i -|1|
            a Cxn matrix
            'S': S, # Collection of sets of indices: S[i] is S_i in the paper - dict indices
            'c': c, # Coefficients of the threshold functions - weights
        }
        """
        self.params = params
        _keys = ['n', 'C', 'b', 'w', 'S', 'c']
        for key in _keys:
            if key not in params:
                raise Exception(f'Key {key} is required.')  # Sanity check - replace assert
        for k, v in params.items():
            setattr(self, k, v)

    def refresh_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)  # Refresh the parameters

    def eval(self, x):
        x = np.array(x)
        obj = 0
        for i in self.C:
            w = np.array(self.w[i])
            obj += self.c[i] * np.min([sum(x[self.S[i]] * w[self.S[i]]), self.b[i]])
        return obj  # Evaluate the function

    def supergradient(self, x):
        x = np.array(x)
        is_not_saturated = [np.sum(x[self.S[i]] * self.w[i][self.S[i]]) <= self.b[i] for i in self.C]
        return np.array(
            [np.sum([self.c[i] * self.w[i][k] * int(k in self.S[i] and is_not_saturated[i]) for i in self.C]) for k in
             range(x.size)])  # Evaluate supergradient


class ZeroOneDecisionSet:
    def __init__(self, n, gamma=0, sigma=0):  # requires problem dimension n
        self.n = n
        self.gamma = gamma
        self.sigma = sigma
        x = cp.Variable(n)  # x >=0 by default
        self.constraints = [x <= 1 - 2 * sigma * n + sigma, x >= sigma]  # x<= 1 by default
        y_param = cp.Parameter(
            n)  # define (unfeasible) point that you wish to project as a parameter of the projection problem

        self.y_param = y_param  # Store unfeasible  variable
        self.x = x  # Store decision variable
        self.euclidean_prob = None
        self.bregman_prob = None

    def setup_constraints(self, additional_constraints):
        # Each time additional constraints are introduced redefine the problem. Usually setup_constraints should only
        # be called in the constructor method.
        self.constraints += additional_constraints

    def project_euclidean(self, y, warm_start=True):  # perform euclidean projection
        if self.euclidean_prob is None:
            self.euclidean_obj = cp.Minimize(
                cp.sum_squares(
                    self.x - self.y_param))  # Euclidean projection finds the closest point in the set to unfeasible y
            self.euclidean_prob = cp.Problem(self.euclidean_obj, self.constraints)  # Store problem instance
        self.y_param.value = y  # replace parameter value
        self.euclidean_prob.solve(
            warm_start=warm_start)  # Solve with warm start. When y is not that different from the previous one,
        # it should be faster.
        return self.x.value  # return projected point

    def project_bregman(self, y, warm_start=False):
        if self.bregman_prob is None:
            divergence = cp.sum(- cp.entr(self.x + self.gamma)) + cp.sum(
                - cp.multiply((self.x + self.gamma), cp.log(self.y_param + self.gamma)))
            self.bregman_obj = cp.Minimize(
                divergence)  # Euclidean projection finds the closest point in the set to unfeasible y
            self.bregman_prob = cp.Problem(self.bregman_obj, self.constraints)  # Store problem instance
        self.y_param.value = y  # replace parameter value
        self.bregman_prob.solve(
            warm_start=warm_start)  # Solve with warm start. When y is not that different from the previous one,
        # it should be faster.
        return self.x.value  # return projected point


class RelaxedPartitionMatroid(ZeroOneDecisionSet):
    def __init__(self, n, cardinalities_k, sets_S, gamma=0, sigma=0):
        super().__init__(n, gamma, sigma)

        self.sets_S = sets_S
        self.cardinalities_k = cardinalities_k
        self.setup_constraints([cp.sum(self.x[sets_S[i]]) == (cardinalities_k[i]) * (
                1 - 2 * self.sigma * self.n) + self.sigma * len(sets_S[i]) for i in range(
            len(cardinalities_k))])  # add additional constraints
        # and inherit functionality from [0, 1] decision set


class OCOPolicy(ABC):
    @abstractmethod
    def __init__(self, decision_set: ZeroOneDecisionSet, objective: ThresholdObjective, eta: float):
        self.eta = eta
        self.objective = objective  # Requires objective
        self.decision_set = decision_set  # Requires decision set #y
        self.frac_rewards = []
        self.int_rewards = []
        self.decision = np.zeros(decision_set.n)
        self.decisions = []
        self.running_time = []
        self.current_iteration = 1
        if isinstance(decision_set, RelaxedPartitionMatroid):
            self.decision = np.zeros(decision_set.n)
            for S, k in zip(decision_set.sets_S, decision_set.cardinalities_k):
                self.decision[S] = k / len(S)
            self.decision = self.decision_set.project_euclidean(self.decision)
        else:
            raise Exception('Not implemented')

    def step(self):
        start = time()
        print(f"Threshold Objective is {self.objective.params}")
        print(f"decision is {self.decision}")
        frac_reward = self.objective.eval(self.decision)
        self.decisions.append(self.decision)
        int_reward = self.objective.eval(self.round(self.decision))
        self.frac_rewards.append(frac_reward)  # Collect value of fractional reward
        self.int_rewards.append(int_reward)  # Collect value of integral reward
        self.current_iteration += 1
        self.running_time.append(time() - start)

    def round(self, x):
        if isinstance(self.decision_set, RelaxedPartitionMatroid):
            return partition_matroid_round(np.copy(x), self.decision_set.sets_S)
        else:
            raise Exception(f'Rounding procedure for {type(self.decision_set)} is not implemented.')


class OGA(OCOPolicy):
    def __init__(self, decision_set: ZeroOneDecisionSet, objective: ThresholdObjective, eta: float, bandit=False):
        super().__init__(decision_set, objective, eta)
        # if decision_set.sigma == 0 and bandit:
        #     raise Exception('Bandit setting cannot run with a nil exploration parameter')
        self.bandit = bandit

    def step(self, eta=None, decision=None, supergradient=None):
        super().step()
        if self.bandit:
            u = sample_spherical(self.decision_set.n)
            self.supergradient = u * self.objective.eval(self.decision + self.decision_set.sigma * u)
        else:
            self.supergradient = self.objective.supergradient(self.decision)  # Compute supergradient

        eta = self.eta if eta is None else eta
        decision = self.decision if decision is None else decision
        supergradient = self.supergradient if supergradient is None else supergradient
        self.decision = self.decision_set.project_euclidean(
            decision + eta * supergradient)  # Take gradient step


class ShiftedNegativeEntropyOMD(OCOPolicy):
    def __init__(self, decision_set: ZeroOneDecisionSet, objective: ThresholdObjective, eta: float):
        super().__init__(decision_set, objective, eta)

    def step(self, eta=None, decision=None, supergradient=None):
        super().step()
        self.supergradient = self.objective.supergradient(self.decision)  # Compute supergradient
        eta = self.eta if eta is None else eta
        decision = self.decision if decision is None else decision
        supergradient = self.supergradient if supergradient is None else supergradient
        self.decision_z = (self.decision + self.decision_set.gamma) * np.exp(
            eta * self.supergradient) - self.decision_set.gamma
        self.decision = self.decision_set.project_bregman(self.decision_z)  # Take gradient step


class OptimisticPolicy(OCOPolicy):
    def __init__(self, decision_set: ZeroOneDecisionSet, objective: ThresholdObjective, Policy: Type[OCOPolicy],
                 eta: float):
        super().__init__(decision_set, objective, eta)
        self.policyPrimary = Policy(decision_set, objective, eta=eta)
        self.policySecondary = Policy(decision_set, objective, eta=eta)

    def step(self, eta=None, pred=None):
        self.supergradient = self.objective.supergradient(self.decision)  # Compute supergradient
        super().step()
        self.policySecondary.step(eta=eta, supergradient=self.supergradient)
        self.policyPrimary.step(eta=eta, decision=np.copy(self.policySecondary.decision), supergradient=np.array(pred))
        self.decision = self.policyPrimary.decision
