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
        self.observation_space = spaces.Box(low=0, high=2, shape=(GAME_LENGTH, WORD_LENGTH)) #Check dtype=int

    def reset(self):
        """
        Reset the environment.
        """
        self.guesses = []
        self.solution_word = random.choice(valid_words.words)
        self.solution = word_to_action(self.solution_word)
        self.board = np.full((GAME_LENGTH, WORD_LENGTH), -1)
        return self.board

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

        self.board[len(self.guesses)] = self._check_guess(self.solution, action)

        if(np.all(self.board[len(self.guesses)] == 2)):
            reward = 1
            done = True
        elif(len(self.guesses) == GAME_LENGTH - 1):
            reward = -1
            done = True
        else:
            reward = 0

        self.guesses.append(action)

        return(self.board, reward, done, {})

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
        print("==============================")
        print()
    
