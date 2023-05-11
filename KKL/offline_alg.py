import numpy as np
import logging
import time
from ContinuousGreedy import ContinuousGreedy, LinearSolver, SamplerEstimator
from KKL.mapping import Mapping

class ApproxAlgorithm:
    def solve(self, w:np.ndarray) -> np.ndarray:
        pass

class ApproxAlgInterval(ApproxAlgorithm):
    def solve(self, w: np.ndarray) -> np.ndarray:
        return np.array([1])

class ApproxGreedy:
    def __init__(self, linear_solver:LinearSolver, mapping:Mapping, initial_point, n:int, 
                alg='continuous-greedy', setting='full-information'):
        self.n = n
        self.iterations = 20
        self.num_of_samples = 10
        self.initial_point = initial_point
        self.linear_solver = linear_solver
        self.mapping = mapping
        self.alg = alg
        self.setting = setting

    def solve(self, w: np.ndarray) -> np.ndarray:
        if self.alg == 'continuous-greedy':
            objective_func = lambda x: np.dot(self.mapping.Phi(x), w)
            estimator = SamplerEstimator(objective_func, self.num_of_samples)
            continuous_greedy = ContinuousGreedy(self.linear_solver, estimator, self.initial_point)
            y = continuous_greedy.fw(self.iterations)
            # round y
            x = np.zeros(self.n)
            for i in range(self.n):
                x[i] = y[i]
            return x
        return self.greedy(w)

    def greedy(self, w: np.ndarray):
        x = np.zeros(self.n)
        unchosen = set(range(self.n))
        f = lambda x: np.dot(self.mapping.Phi(x), w)
        for _ in range(self.linear_solver.k):
            max = -1e15
            max_elem = None
            for i in unchosen:
                elem = np.zeros(self.n)
                elem[i] = 1
                marginal_gain = f(x + elem) - f(x)
                assert marginal_gain >= 0, "marginal gain < 0"
                if marginal_gain >= max:
                    max = marginal_gain
                    max_elem = i

            x[max_elem] = 1
            unchosen.remove(max_elem)

        assert sum(x) == self.linear_solver.k, "greedy algorithm problem"
        assert np.all(x >= -0.001), f"x = {x} < 0"
        assert np.all(x <= 1.001), f"x = {x} > 1" 
        return x
