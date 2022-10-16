import gym
from gym import spaces
import random
import numpy as np
from colorama import Style

from ..data import valid_words
from ..data import colors
from ..utils import action_to_word, word_to_action
from ..config import WORD_LENGTH, GAME_LENGTH, WORDS_FILE

class QWordle(gym.Env):
    """
    Class to implement a gym environment for Wordle.
    """

    def __init__(self):
        """
        Initialize the environment.
        """
        self.action_space = spaces.MultiDiscrete([26] * WORD_LENGTH)
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=-1, high=2, shape=(GAME_LENGTH, WORD_LENGTH), dtype=np.int8),
            'letters': spaces.Box(low=-1, high=2, shape=(26,), dtype=np.int8)
        })

    def reset(self):
        """
        Reset the environment.
        """
        self.guesses = []
        self.solution_word = random.choice(valid_words.words)
        self.solution = word_to_action(self.solution_word)
        self.board = np.full((GAME_LENGTH, WORD_LENGTH), -1)
        self.letters = np.full((26,), -1)
        return {'board': self.board, 'letters': self.letters}

    def _check_guess(self, solution, pred):
        """
        Check if guess is correct.
        """
        return [(2 if x == solution[i] else ( 1 if x in solution else 0)) for i, x in enumerate(pred)]

    def step(self, action):
        """
        Perform a step in the environment.
        """
        reward = None
        done = False

        res = self._check_guess(self.solution, action)
        self.board[len(self.guesses)] = res

        for i, x in enumerate(res):
            if x == 2:
                self.letters[action[i]] = 2
            elif x == 1:
                self.letters[action[i]] = 1
            else:
                self.letters[action[i]] = 0

        if(np.all(self.board[len(self.guesses)] == 2)):
            reward = 1
            done = True
        elif(len(self.guesses) == GAME_LENGTH - 1):
            reward = -1
            done = True
        else:
            reward = 0

        self.guesses.append(action)

        return({'board': self.board, 'letters': self.letters}, reward, done, {})

    def render(self):
        """
        Render the environment.
        """
        print("++++++++++++++++++++++++++++++")
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
    
