import itertools

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

class ThresholdObjective:
    def __init__(self, params):
        """Params format:
        params = { The naming corresponds to the one used in the paper
            'n': n, # dimension size - n, |V|
            'C': C, # Number of threshold functions - # of wdnfs
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
    def get_opt(self, decision_set):
        x = decision_set.x
        obj = 0
        for i in self.C:
            w = np.array(self.w[i])
            obj += self.c[i] * cp.min(cp.hstack([cp.sum(x[self.S[i]] * w[self.S[i]]), self.b[i]]))
        problem = cp.Problem(cp.Maximize(obj), constraints=decision_set.constraints)
        problem.solve()
        return obj.value, x.value


    def supergradient(self, x):
        x = np.array(x)
        is_not_saturated = [np.sum(x[self.S[i]] * np.array(self.w[i])[self.S[i]]) <= self.b[i] for i in self.C]
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
        self.support = np.arange(n)

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
            divergence = cp.sum(- cp.entr(self.x[self.support] + self.gamma)) + cp.sum(
                - cp.multiply((self.x[self.support] + self.gamma), cp.log(self.y_param[self.support] + self.gamma)))
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
        self.support = np.unique(reduce(concat, sets_S))

class OCOPolicy:
    def __init__(self, decision_set: ZeroOneDecisionSet, objective: ThresholdObjective, eta: float):
        self.eta = eta
        self.objective = objective  # Requires objective
        self.decision_set = decision_set  # Requires decision set #y
        self.frac_rewards = []
        self.int_rewards = []
        self.decision = np.ones(decision_set.n)
        self.decisions = []
        self.int_decisions = []
        self.running_time = []
        self.current_iteration = 0
        if isinstance(decision_set, RelaxedPartitionMatroid):
            self.decision = np.ones  (decision_set.n)
            for S, k in zip(decision_set.sets_S, decision_set.cardinalities_k):
                self.decision[S] = k / len(S)
            self.decision = self.decision_set.project_euclidean(self.decision)
        else:
            raise Exception('Not implemented')

    def step(self):
        start = time()
        # print(f"Threshold Objective is {self.objective.params}")
        # print(f"decision is {self.decision}")
        frac_reward = self.objective.eval(self.decision)
        self.decisions.append(self.decision)
        int_decision = self.round(self.decision)
        self.int_decisions.append(int_decision)
        int_reward = self.objective.eval(int_decision)
        
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
    def __init__(self, decision_set: ZeroOneDecisionSet, objective: ThresholdObjective, eta: float, gamma: float):
        super().__init__(decision_set, objective, eta)

        self.gamma = gamma

    def step(self, eta=None, decision=None, supergradient=None):
        super().step()
        self.supergradient = self.objective.supergradient(self.decision)  # Compute supergradient
        eta = self.eta if eta is None else eta
        decision = self.decision if decision is None else decision
        supergradient = self.supergradient if supergradient is None else supergradient
        self.decision_z = (self.decision + self.gamma) * np.exp(
            eta * supergradient) - self.gamma
        # self.decision =  self.decision_set.project_bregman(self.decision_z)  # Take gradient step
        self.decision = self.decision_set.project_euclidean(self.decision_z)

class MetaPolicy(OCOPolicy):
    def __init__(self, decision_set: ZeroOneDecisionSet, objective: ThresholdObjective, Policy: Type[OCOPolicy],
                 etas: List[float]):
        super().__init__(decision_set, objective, np.nan)
        self.experts = [Policy(decision_set, objective, eta=eta) for eta in etas]
        simplex = RelaxedPartitionMatroid(len(etas), cardinalities_k=[1], sets_S=[list(range(len(etas)))])
        self.policy = Policy(simplex, objective, eta=np.nan)
        self.sum_grads = np.zeros(len(etas))

    def step(self, eta=None, pred=None):
        self.decision = np.sum([self.experts[i].decision * self.policy.decision[i] for i in range(len(self.experts))],
                               axis=0)
        super().step()
        gradient = []
        for expert in self.experts:
            gradient.append(self.objective.eval(expert.decision))
            expert.objective = self.objective
            expert.step()
        gradient = np.array(gradient)
        self.sum_grads += np.linalg.norm(gradient, 2) ** 2

        eta = 1 / np.sqrt(self.sum_grads)
        # self.policy.step(eta)
        self.policy.decision = self.policy.decision_set.project_euclidean(
            self.policy.decision + eta * gradient)  # Take gradient step
        self.policy.decisions.append(self.policy.decision)


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


def in_circle(circle_x, circle_y, circle_rad, x, y):
    return ((x - circle_x) * (x - circle_x) + (y - circle_y) * (y - circle_y) <= circle_rad * circle_rad)


class FixedShare(OCOPolicy):
    def __init__(self, decision_set: ZeroOneDecisionSet, objective: ThresholdObjective, eta: float, beta: float):
        super().__init__(decision_set, objective, eta)
        self.beta = beta

    def step(self, eta=None, decision=None, supergradient=None):
        # self.supergradient = self.objective.supergradient(self.decision)  # Compute supergradient
        eta = self.eta if eta is None else eta
        v = self.decision * np.exp(eta * supergradient)
        self.decision = self.beta * np.sum(v) / v.size + (1 - self.beta) * v
        self.decision = self.decision / sum(self.decision)


class EXP3:
    def __init__(self, n: int, eta: float):
        self.eta = eta
        self.n = n
        self.w = np.ones(n) / n
        self.sample()

    def step(self, bandit_reward=None):
        eta = self.eta
        l = np.zeros(self.n)
        l[self.action] = bandit_reward / self.get_distibution()[self.action]
        self.w = self.w * np.exp(eta * l / self.n)
        self.sample()

    def sample(self):
        self.action = np.random.choice(np.arange(self.n), p=self.get_distibution())

    def get_distibution(self):
        return self.eta * 1 / self.n + (1 - self.eta) * self.w / sum(self.w)


class FSF(OCOPolicy):
    def __init__(self, decision_set: ZeroOneDecisionSet, objective: ThresholdObjective,
                 eta: float, gamma: float):
        super().__init__(decision_set, objective, eta)
        self.k = decision_set.cardinalities_k[0]
        self.n = decision_set.n
        simplex = RelaxedPartitionMatroid(decision_set.n, cardinalities_k=[1], sets_S=[np.arange(decision_set.n)])
        self.experts = [FixedShare(simplex, objective, eta=eta, beta=gamma) for _ in range(self.k)]

    def step(self, eta=None):
        x = np.zeros((self.k, self.n))
        for k, expert in enumerate(self.experts):
            p = expert.decision
            p[p < 0] = 0
            p = p / sum(p)  # numerical stability
            p_i = np.random.choice(np.arange(p.size), p=p)
            if k > 0:
                x[k] = np.copy(x[k - 1])
            x[k][p_i] = 1

        self.decision = np.copy(x[-1, :])
        super().step()
        for k, expert in enumerate(self.experts):
            l = np.zeros(self.n)
            for i in range(self.n):
                xs = x[k - 1] if k > 0 else np.zeros(self.n)

                xsp = np.copy(xs)
                xsp[i] = 1
                l[i] = self.objective.eval(xsp) - self.objective.eval(xs)
            expert.step(supergradient=l)


class OnlineTBG(OCOPolicy):
    def __init__(self,decision_set:ZeroOneDecisionSet,  objective: ThresholdObjective, eta: float, n_colors: int):
        super().__init__(decision_set, objective, eta)
        # self.experts = {}
        self.experts = {}
        self.items = []
        for i, k in enumerate(decision_set.cardinalities_k):
            self.items.extend([decision_set.sets_S[i]]*k)
        self.n_slots = sum(decision_set.cardinalities_k)
        self.n_colors = n_colors
        self.n = self.decision_set.n
        self.objective = objective
        for slot in range(self.n_slots):
            for c in range(n_colors):
                simplex = RelaxedPartitionMatroid(len(self.items[slot]), cardinalities_k=[1],
                                                  sets_S=[np.arange(len(self.items[slot]))])
                simplex.S = self.items[slot]
                self.experts[(slot, c)] = FixedShare(simplex, objective, eta=eta, beta=0)  # Hedge

    def step(self, eta=None):
        global_action = {}
        for slot in range(self.n_slots):
            for c in range(self.n_colors):
                global_action[(slot, c)] = np.random.choice(self.items[slot], p=self.experts[(slot, c)].decision)
        c_vec = np.zeros(self.n_slots)
        for c in range(self.n_slots):
            c_vec[c] = np.random.choice(np.arange(self.n_colors))
        G, G_vec = self.sample(global_action, c_vec)
        self.decision = np.copy(G_vec)
        super().step()
        for slot in range(self.n_slots):
            for c in range(self.n_colors):
                Gp = {}
                for slotp in range(self.n_slots):
                    for cp in range(c):
                        Gp[(slotp, cp)] = global_action[(slotp, cp)]
                for slotp in range(slot):
                    Gp[(slotp, c)] = global_action[(slotp, c)]
                sampled_Gp, sampled_Gp_vec = self.sample(Gp, c_vec)
                feedback = np.zeros(len(self.items[slot]))
                for item in range(len(self.items[slot])):
                    A = np.copy(sampled_Gp_vec)
                    A[item] = 1
                    feedback[item] = self.objective.eval(A)
                self.experts[(slot, c)].step(supergradient=feedback)

    def sample(self, global_action, c_vec):
        G = {}
        for slot, color in global_action:
            if color == c_vec[slot]:
                G[slot] = global_action[(slot, color)]
        G_vec = np.zeros(self.n)
        for slot in G:
            G_vec[G[slot]] = 1
        G_vec[~np.isin(np.arange(self.decision_set.n), decision_set.support)] =1
        return G, G_vec


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
    return objectives


if __name__ == "__main__":

    # Generate coverage example
    objectives = generate_non_stationary_problem()
    n = objectives[0].n
    decision_set = RelaxedPartitionMatroid(n, cardinalities_k=[2, 1, 2,3], sets_S=[[1,2,3,4], [5,6], [7,8,9], list(range(10,20,1))])
    # decision_set = RelaxedPartitionMatroid(n, cardinalities_k=[5], sets_S=[list(range(0,20,1))])

    policyTBG = OnlineTBG(decision_set=decision_set,objective=objectives[0], eta=1, n_colors=1)
    policyFSF = FSF(decision_set=decision_set,objective=objectives[0], eta=.05, gamma = .001)
    policyOMD = ShiftedNegativeEntropyOMD(decision_set=decision_set,objective=objectives[0], eta=.01, gamma=0.0)
    # policyShiftedOMD = ShiftedNegativeEntropyOMD(decision_set=decision_set, objective=objectives[0], eta=.05, gamma=0.02)
    policyOGA = OGA(decision_set=decision_set, objective=objectives[0], eta=.001)
    def run_non_stationary_exp(policy, name, T_half = 100):
        for t in range(T_half):
            policy.step()
            print(policyTBG.decision, sum(policyTBG.decision))

        plt.plot(taverage(policy.frac_rewards), label = name)
    # run_non_stationary_exp(policyOMD, 'OMD')
    # run_non_stationary_exp(policyShiftedOMD, 'SOMD')
    # run_non_stationary_exp(policyOGA, 'OGA')
    run_non_stationary_exp(policyTBG, 'TBG')
    # run_non_stationary_exp(policyFSF, 'FSF')
    plt.plot([0,100], [objectives[0].get_opt(decision_set)[0]]*2, linestyle = '--', color = 'black')
    plt.legend()
    plt.show()

    # print(policyOGA.decision,sum(policyOGA.decision))
