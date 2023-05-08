import random
import numpy as np
from KKL.mapping import WDNFMapping
from KKL.offline_alg import ApproxAlgorithm
from typing import Tuple


class KKL:
    def __init__(self, approx_alg:ApproxAlgorithm, 
                mapping:WDNFMapping, alpha:float, delta:float, eta:float, R:float, n:int):
        self.n = n
        self.R = R
        self.eta = eta
        self.delta = delta
        self.alpha = alpha
        self.lambda_ = delta / (4 * (self.alpha + 2)**2 * self.R**2)
        self.approx_alg = approx_alg
        self.mapping = mapping

        self.s = np.zeros(self.n)
        self.x = self.mapping.Phi(self.s)

    def play(self, history):
        ''' Given the history so far, returns the next action '''
        if not history: # first action
            return self.s

        feedback, w = history[-1]
        y = self.x + self.eta * w
        self.s, self.x = self.approx_proj(y)
        
        return self.s

    def approx_proj(self, z:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ''' APPROX-PROJ algorithm '''
        s = np.zeros(self.n)
        x = self.mapping.Phi(s)
        while True:
            ss, xx = self.extended_approx_oracle(x - z)
            if np.dot(x, x - z) <= np.dot(xx, x - z) + self.delta:
                return s, x
            x = self.lambda_ * xx + (1 - self.lambda_) * x
            s = ss if random.uniform(0, 1) <= self.lambda_ else s
        

    def extended_approx_oracle(self, v:np.ndarray) -> np.ndarray:
        '''
            extended approximation oracle
        '''
        s = self.approx_alg.solve(self.project_w(v))
        vec =  self.project_w(v) - v
        norm = np.linalg.norm(vec)
        phi_s = self.mapping.Phi(s)
        x = phi_s + self.R * (self.alpha + 1) * vec / norm if norm > 0 else phi_s
        return s, x   

    def project_w(self, x:np.ndarray) -> np.ndarray:
        ''' 
            The set W of feasible decision is [0,1]^n in our experiments.
            Thus, W+ = {aw | w \in W, a >= 0 } is the positive orthant.
            Since our problems are maximization problems we need to project to 
            -W+, i.e. W- = {aw | w \in W, a <= 0}.
            W- is the negative orthant.
            For more info, see "Playing games with approximation algorithm [KKL]"
        '''
        return np.minimum(x, np.zeros(len(x)))


    