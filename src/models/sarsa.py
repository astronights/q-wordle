from .base_model import BaseModel
from ..strategies.random_strategy import RandomStrategy
from ..strategies.highest_ll_strategy import HighestLLStrategy
from ..strategies.highest_ll_smart_strategy import HighestLLSmartStrategy
from ..strategies.fresh_letters_strategy import FreshLettersStrategy
from ..envs.qwordle import QWordle
from ..utils import get_state, word_to_action
from ..config import WORD_LENGTH, GAME_LENGTH

import os
import pickle

import numpy as np
from tqdm import tqdm

class SARSALearn(BaseModel):
    
    def __init__(self, config = None):
        super().__init__(config)
        self.env = QWordle()
        self.strategy_len = len(self.env.strategies)
        if 'Q' in config:
            self.Q = config['Q']
        else:
            self.Q = np.zeros((WORD_LENGTH+1, WORD_LENGTH+1, GAME_LENGTH+1, self.strategy_len))
        self.epsilon = config['epsilon']
        self.gamma = config['gamma']
        self.alpha = config['alpha']
        self.games_solved = []

    def update_q(self, q):
        self.Q = q

    def policyFunction(self, state, epsilon):
        action_probabilities = np.ones(self.strategy_len, dtype = float) * epsilon / self.strategy_len   
        best_action = np.argmax(self.Q[state[0], state[1], state[2], :])
        action_probabilities[best_action] += (1.0 - epsilon)
        return action_probabilities
   
    def train(self, iter = 100):
        self.games_solved = []
        num_solved = 0
        for i in tqdm(range(iter)):
            state = self.env.reset()
            done = False
            while(not done):
                action_probabilities = self.policyFunction(state, self.epsilon*(1 - num_solved/iter))
                action_strategy = np.random.choice(np.arange(len(action_probabilities)), p = action_probabilities)
                next, reward, done, res = self.env.step(action_strategy)

                next_action_probabilities = self.policyFunction(next, self.epsilon*(1 - num_solved/iter))
                next_action_strategy = np.random.choice(np.arange(len(next_action_probabilities)), p = next_action_probabilities)

                q_target = reward + self.gamma * self.Q[next[0], next[1], next[2], next_action_strategy]
                self.Q[state[0], state[1], state[2], action_strategy] = (self.alpha*q_target) + ((1-self.alpha) * self.Q[state[0], state[1], state[2], action_strategy])
                state = next
            if(res['solved']):
                num_solved += 1
                self.games_solved.append(i+1)

        pickle.dump({'Q': self.Q}, open('sarsa.pkl', 'wb'))

    def test(self, verbose=True):
        if(os.path.exists('sarsa.pkl')):
            self.Q = pickle.load(open('sarsa.pkl', 'rb'))['Q']
        state = self.env.reset()
        done = False
        while(not done):
            action_probabilities = self.policyFunction(state, 0)
            action_strategy = np.random.choice(np.arange(len(action_probabilities)), p = action_probabilities)
            next, _, done, res = self.env.step(action_strategy)
            state = next
            if(verbose):
                self.env.render()
        if(verbose):
            print(res)
        return res