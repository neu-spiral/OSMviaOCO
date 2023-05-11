import numpy as np
from collections import deque
from KKL.online_alg import KKL
from KKL.mapping import Mapping
from typing import List, Tuple


class Game:
    def __init__(self, alg:KKL, mapping:Mapping, ws: List[np.ndarray], n:int, horizon:int):
        """

        :type n: object
        """
        self.algorithm = alg    # online algorithm
        self.ws = ws            # list of "weight" vectors
        self.horizon = horizon  # T
        self.n = n              # dimension of the decision/action
        self.mapping = mapping  # Phi(s)

        self.history = deque()
        self.reward_history = deque()
        self.action_history = deque()
        self.timestep = 0

    def next(self) -> Tuple[np.ndarray, float, np.ndarray]:
        ''' 
            play the next action, receive feedback, update history
            Output: action, feedback, w
        '''
        action = self.algorithm.play(self.history)
    
        assert len(action) == self.n, f"dimension of actions is {len(action)} != {self.n}"
        assert np.all(action >= -0.001), "action < 0"
        assert np.all(action <= 1.001), "s > 1"
    
        w = self.ws[self.timestep] # get w_t
        feedback = np.dot(self.mapping.Phi(action), w)

        assert feedback >= -0.001 , f"feedback < 0"

        # update history
        self.history.append((feedback, w))
        
        self.timestep += 1

        return action, feedback, w

    def play(self):
        ''' play all rounds '''
        for t in range(self.horizon):
            action, feedback, w = self.next()
            self.reward_history.append(feedback)
            self.action_history.append(action)
            print(f"iteration {t}")
            print(f"cum avg reward {self.get_cum_avg_reward()[-1]}")
            print(f"reward #{t} {feedback}")
            print(f"action #{t} {action}")

    def get_reward_history(self) -> np.ndarray:
        return np.array(self.reward_history)

    def get_action_history(self) -> np.ndarray:
        return np.array(self.action_history)

    def get_cum_avg_reward(self) -> np.ndarray:
        return np.cumsum(self.reward_history) / (np.arange(len(self.reward_history)) + 1)