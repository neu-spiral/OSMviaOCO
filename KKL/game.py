import numpy as np
from collections import deque
from online_alg import KKL
from mapping import Mapping

class Game:
    def __init__(self, alg:KKL, mapping:Mapping, ws:list[np.ndarray], n:int, horizon:int):
        self.algorithm = alg    # online algorithm
        self.ws = ws            # list of "weight" vectors
        self.horizon = horizon  # T
        self.n = n              # dimension of the decision/action
        self.mapping = mapping  # Phi(s)


        self.history = deque()
        self.reward_history = deque()
        self.action_history = deque()
        self.timestep = 0

    def next(self) -> tuple[np.ndarray, float, np.ndarray]:
        ''' 
            play the next action, receive feedback, update history
            Output: action, feedback, w
        '''
        action = self.algorithm.play(self.history)

        if not self.isValid(action):
            raise Exception(f"Game play method, action s = {action} is not a valid action")
        
        w = self.ws[self.timestep] # get w_t
        feedback = np.dot(self.mapping.Phi(action), w)

        # update history
        self.history.append((feedback, w))
        
        self.timestep += 1

        return action, feedback, w
    
    
    def isValid(self, s:np.ndarray) -> bool:
        ''' check that dimensions match '''
        return  len(s) == self.n
            

    def play(self):
        ''' play all rounds starting '''
        print_interval = self.horizon * 0.1
        for t in range(self.horizon):
            action, feedback, w = self.next()
            self.reward_history.append(feedback)
            self.action_history.append(action)
            if t % print_interval == 0:
                print(f"Progress: {t / self.horizon}")

    def get_reward_history(self) -> np.ndarray:
        return np.array(self.reward_history)

    def get_action_history(self) -> np.ndarray:
        return np.array(self.action_history)

    def get_cum_avg_reward(self) -> np.ndarray:
        return np.cumsum(self.reward_history) / (np.arange(self.horizon) + 1)