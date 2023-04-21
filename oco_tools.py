import numpy as np
import cvxpy as cp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


class ThresholdObjective:
    def __init__(self, params):
        """Params format:
        params = { The naming corresponds to the one used in the paper
            'n': n, # dimension size - n, |V|
            'C': C, # Number of threshold functions - # of wdnfs
            'b': b, # Vector of thresholds - |1|_C
            'w': w, # List of vectors (variable size) w[i][j] is the weight of the x[j] within threshold function i -|1|
            'S': S, # Collection of sets of indices: S[i] is S_i in the paper - dict indices
            'c': c, # Coefficients of the threshold functions - weights
        }
        """
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
        is_saturated = [np.sum(x[self.S[i]]) <= self.b[i] for i in self.C]
        return np.array(
            [np.sum([self.w[i][k] * int(k in self.S[i] and is_saturated[i]) for i in self.C]) for k in
             x.size])  # Evaluate supergradient


class ZeroOneDecisionSet:
    def __init__(self, n):  # requires problem dimension n
        x = cp.Variable(n, nonneg=True)  # x >=0 by default
        self._default_constraint = [x <= 1]  # x<= 1 by default
        y_param = cp.Parameter(
            n)  # define (unfeasible) point that you wish to project as a parameter of the projection problem
        self.obj = cp.Minimize(
            cp.sum_squares(x - y_param))  # Euclidean projection finds the closest point in the set to unfeasible y
        self.y_param = y_param  # Store unfeasible  variable
        self.x = x  # Store decision variable
        self.prob = cp.Problem(self.obj, [self._default_constraint])  # Store problem instance

    def setup_constraints(self, additional_constraints):
        # Each time additional constraints are introduced redefine the problem. Usually setup_constraints should only
        # be called in the constructor method.
        prob = cp.Problem(self.obj, [self._default_constraint] + additional_constraints)
        self.prob = prob

    def project_euclidean(self, y, warm_start=True):  # perform euclidean projection
        self.y_param.value = y  # replace parameter value
        self.prob.solve(
            warm_start=warm_start)  # Solve with warm start. When y is not that different from the previous one,
        # it should be faster.
        return self.x.value  # return projected point
    # other projection schemes will be added


class RelaxedPartitionMatroid(ZeroOneDecisionSet):
    def __init__(self, n, cardinalities_k, sets_S):
        super().__init__(n)
        self.setup_constraints([cp.sum(self.x[sets_S[i]]) <= cardinalities_k[i] for i in range(
            len(cardinalities_k))])  # add additional constraints and inherit functionality from [0, 1] decision set


        
        
        
class OCOPolicy:
    def __init__(self, decision_set: ZeroOneDecisionSet, eta: float, objective: ThresholdObjective):
        self.eta = eta
        self.objective = objective  # Requires objective
        self.decision_set = decision_set  # Requires decision set #y
        self.frac_rewards = []
        self.int_rewards = []
        self.decision = np.zeros(decision_set.n)
        self.current_iteration = 1

    def step(self, eta=None):
        if eta is None:
            eta = self.eta  # Set time varying learning rate
        frac_reward = self.objective.eval(self.decision) #what is self.decision
        int_reward = self.objective.eval(self.round(self.decision))
        supergradient = self.objective.supergradient(self.decision)  # Compute supergradient
        self.frac_rewards.append(frac_reward)  # Collect value of fractional reward
        self.int_rewards.append(int_reward)  # Collect value of integral reward
        self.decision = self.decision_set.project_euclidean(self.decision + eta * supergradient)  # Take gradient step
        self.current_iteration += 1

    def round(self, x):  # Add appropriate rounding
        return x
