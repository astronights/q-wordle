from ctypes.wintypes import WORD
from .base_strategy import BaseStrategy
from ..data import valid_words, word_ll
from ..utils import action_to_word, invert_dict, word_to_action
from ..config import WORD_LENGTH

import numpy as np
import random

class HighestLLSmartStrategy(BaseStrategy):

    def __init__(self, config = None):
        super().__init__(config)
        self.word_ll = word_ll.word_ll
        self.results = {'green': {}, 'yellow': {}}
        self.green = np.full(WORD_LENGTH, -1)
        self.yel_loc = {}

    def reset(self):
        self.results = {'green': {}, 'yellow': {}}
        self.yel_loc = {}
        self.green = np.full(WORD_LENGTH, -1)

    def get_action(self, observations):
        self.reset()
        for gi, guess in enumerate(observations['guesses']):
            mgy = observations['board'][gi]
            for i, color in enumerate(mgy):
                if(color == 2):
                    if(i not in self.results['green']):
                        self.results['green'][i] = guess[i]
                elif(color == 1):
                    if(i not in self.results['yellow']):
                        self.results['yellow'][i] = set([guess[i]])
                    else:
                        self.results['yellow'][i].add(guess[i])
                    if(guess[i] not in self.yel_loc):
                        self.yel_loc[guess[i]] = set([i])
                    else:
                        self.yel_loc[guess[i]].add(i)
                else:
                    pass
        observed_words = [''.join(action_to_word(x)).lower() for x in observations['guesses']]
        fresh_words = {k:v for k, v in self.word_ll.items() if k not in observed_words}
        if(len(observations['guesses'])==0 or (len(self.results['green'])==0) and len(self.results['yellow'])==0):
            return max(fresh_words, key=fresh_words.get)
        for k, v in self.results['green'].items():
            self.green[k] = v
        filtered_words = dict(filter(self.word_filter, fresh_words.items()))
        return max(filtered_words, key=filtered_words.get)

    def word_filter(self, item):
        action = np.array(word_to_action(item[0]))
        match = np.array_equal(np.where((action - self.green) == 0), np.where(self.green != -1))
        if(match):
            if(len(self.results['yellow']) == 0):
                return True
            else:
                for k, v in self.results['yellow'].items():
                    if(action[k] in v):
                        return False
                for k, v in self.yel_loc.items():
                    remaining_positions = np.setdiff1d(range(WORD_LENGTH), v) #Checking all green positions too cause too complex
                    if(k not in action[remaining_positions]):
                        return False
                return True
        else:
            return False
