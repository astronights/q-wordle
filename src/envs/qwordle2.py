import gym
from gym import spaces
import random
import numpy as np
from colorama import Style

from collections import Counter

from ..data import valid_words, secret_words
from ..data import colors
from ..utils import action_to_word, word_to_action
from ..config import WORD_LENGTH, GAME_LENGTH, WIN_REWARD, LOSE_REWARD, GREEN_REWARD, YELLOW_REWARD, GREY_REWARD

from ..strategies import base_strategy, fresh_letters_strategy, highest_ll_strategy, highest_ll_smart_strategy

class QWordle2(gym.Env):
    """
    Class to implement a gym environment for Wordle.
    """

    def __init__(self):
        """
        Initialize the environment.
        """
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=-1, high=2, shape=(GAME_LENGTH, WORD_LENGTH), dtype=np.int8),
            'guesses': spaces.Box(low=0, high=26, shape=(GAME_LENGTH, WORD_LENGTH), dtype=np.int8),
            'letters': spaces.Box(low=-1, high=2, shape=(26,), dtype=np.int8)
        })

        self.strategies = {
            'fresh_letters': fresh_letters_strategy.FreshLettersStrategy(),
            'highest_ll': highest_ll_strategy.HighestLLStrategy(),
            'highest_ll_smart': highest_ll_smart_strategy.HighestLLSmartStrategy()
        }

    def reset(self):
        """
        Reset the environment.
        """
        self.guesses = []
        self.solution_word = random.choice(secret_words.secret_words)
        self.solution = word_to_action(self.solution_word)
        self.board = np.full((GAME_LENGTH, WORD_LENGTH), -1)
        self.letters = np.full((26,), -1)
        return {'board': self.board, 'guesses': self.guesses, 'letters': self.letters}

    def _check_guess(self, solution, pred):
        """
        Check if guess is correct.
        """
        res = [None] * WORD_LENGTH
        solution_counter = Counter(solution)
        for i, x in enumerate(pred):
            if x == solution[i]:
                solution_counter[x] -= 1
                res[i] = 2
        for i, x in enumerate(pred):
            if(res[i] is None):
                if x in solution:
                    if(solution_counter[x] > 0):
                        res[i] = 1
                    else:
                        res[i] = 0
                    solution_counter[x] -= 1
                else:
                    res[i] = 0
        return res

    def step(self, action):
        """
        Perform a step in the environment.
        """
        reward = None
        done = False
        strategy = self.strategies[action]
        word = strategy.get_action({'board': self.board, 'guesses': self.guesses, 'letters': self.letters})
        word = word_to_action(word)
        res = self._check_guess(self.solution, word)
        self.board[len(self.guesses)] = res

        for i, x in enumerate(res):
            if x == 2:
                self.letters[word[i]] = 2
            elif x == 1:
                if(self.letters[word[i]] < 2):
                    self.letters[word[i]] = 1
            else:
                if(self.letters[word[i]] < 1):
                    self.letters[word[i]] = 0

        solved = False
        if(np.all(self.board[len(self.guesses)] == 2)):
            reward = WIN_REWARD
            done = True
            solved = True
        elif(len(self.guesses) == GAME_LENGTH - 1):
            reward = LOSE_REWARD
            done = True
        else:
            reward = res.count(2)*GREEN_REWARD + res.count(1)*YELLOW_REWARD - res.count(0)*GREY_REWARD

        self.guesses.append(word)

        return({'board': self.board, 'guesses': self.guesses, 'letters': self.letters}, reward, done, {'solved': solved})

    def render(self):
        """
        Render the environment.
        """
        print("++++++++++++++++++++++++++++++")
        print(f"Solution: {self.solution_word}")
        for i, guess in enumerate(self.guesses):
            word = action_to_word(guess)
            for j in range(WORD_LENGTH):
                print(colors.color_map[self.board[i][j]] + Style.BRIGHT + word[j] + ' ', end='')
            print()
        print()
        for i in range(26):
            print(colors.color_map[self.letters[i]] + Style.BRIGHT + chr(ord('A') + i) + ' ', end='')
        print()
        print("==============================")
        print()
    
