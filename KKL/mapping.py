import numpy as np

class Mapping:
    def Phi(self, s: np.ndarray) -> np.ndarray:
        pass

class IdentityMapping(Mapping):
    def Phi(self, s:np.ndarray) -> np.ndarray:
        return s

class WDNFMapping(Mapping):
    def __init__(self, n:int, index_to_set:set, sign=-1):
        self.n = n
        self.m = len(index_to_set)
        self.index_to_set = index_to_set
        self.sign = sign

    def Phi(self, s:np.ndarray) -> np.ndarray:
        ''' Maps a decision s \in R^n to a vector in R^m '''
        phi = np.zeros(self.m)
        for i in range(self.m):
            set = self.index_to_set[i] # returns a tuple
            if self.sign == -1:
                phi[i] = np.prod([1 - s[j] for j in set])
            else:
                phi[i] = np.prod([s[j] for j in set])
        return phi