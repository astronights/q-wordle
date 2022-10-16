from .base_model import BaseModel
from ..strategies.random import RandomStrategy
from ..envs.qwordle import QWordle
from ..utils import get_state, word_to_action

import numpy as np

class QLearn(BaseModel):
    
    def __init__(self, config = None):
        super().__init__(config)
        self.strategies = []
        self.strategies.append(RandomStrategy())
        self.Q = np.zeros((6, 6, 26, len(self.strategies)))
        self.epsilon = 0.1
        self.gamma = 0.6
        self.env = QWordle()

    def policyFunction(self, state):
        action_probabilities = np.ones(len(self.strategies), dtype = float) * self.epsilon / len(self.strategies)       
        best_action = np.argmax(self.Q[state['green'], state['yellow'], state['missing'], :])
        action_probabilities[best_action] += (1.0 - self.epsilon)
        return action_probabilities
   
    def train(self,trial = 100, iter = 100):
        for i in range(iter):
            board, letters = self.env.reset()
            state = get_state(letters) 
            done = False
            while(not done):
                action_probabilities = self.policyFunction(state)
                action_strategy = np.random.choice(np.arange(len(action_probabilities)), p = action_probabilities)
                action = self.strategies[action_strategy].get_action()
                action = word_to_action(action)
                next, reward, done, _ = self.env.step(action)
                next_state = get_state(next['letters'])
                next_best_action = np.argmax(self.Q[next_state['green'], next_state['yellow'], next_state['missing'], :])
                q_target = reward + self.gamma * self.Q[next_state['green'], next_state['yellow'], next_state['missing'], next_best_action]
                self.Q[state['green'], state['yellow'], state['missing'], action_strategy] = q_target -  self.Q[state['green'], state['yellow'], state['missing'], action_strategy]
                state = next_state
            print(self.Q)

    def test(self):
        pass