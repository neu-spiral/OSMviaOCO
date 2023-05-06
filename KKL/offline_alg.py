import sys
sys.path.append('../')

import numpy as np
from ContinuousGreedy import ContinuousGreedy, LinearSolver, SamplerEstimator
from mapping import Mapping

class ApproxAlgorithm:
    def solve(self, w:np.ndarray) -> np.ndarray:
        pass

class ApproxAlgInterval(ApproxAlgorithm):
    def solve(self, w: np.ndarray) -> np.ndarray:
        return np.array([1])

class ApproxGreedy:
    def __init__(self, linear_solver:LinearSolver, mapping:Mapping, initial_point, n:int):
        self.n = n
        self.iterations = 20
        self.num_of_samples = 10
        self.initial_point = initial_point
        self.linear_solver = linear_solver
        self.mapping = mapping

    def solve(self, w: np.ndarray) -> np.ndarray:
        objective_func = lambda x: np.dot(self.mapping.Phi(x), w)
        estimator = SamplerEstimator(objective_func, self.num_of_samples)
        continuous_greedy = ContinuousGreedy(self.linear_solver, estimator, self.initial_point)
        y, _, _ = continuous_greedy.fw(self.iterations)
        # round y
        x = np.zeros(self.n)
        for i in range(self.n):
            x[i] = y[i]
        return x
