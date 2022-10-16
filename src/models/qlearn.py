from .base_model import BaseModel
from ..strategies.random import RandomStrategy

import numpy as np

class QLearn(BaseModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.strategies = []
        self.strategies.append(RandomStrategy())

    def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
        def policyFunction(state):
    
            Action_probabilities = np.ones(num_actions,
                    dtype = float) * epsilon / num_actions
                    
            best_action = np.argmax(Q[state])
            Action_probabilities[best_action] += (1.0 - epsilon)
            return Action_probabilities
   
        return policyFunction
        

    def train(self,trial, iter):
        pass

    def test(self):
        pass