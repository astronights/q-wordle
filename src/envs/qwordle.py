import gym
from gym import spaces
import random
import numpy as np
from colorama import Style

from collections import Counter

from ..data import secret_words
from ..data import colors
from ..utils import action_to_word, word_to_action, get_state
from ..config import WORD_LENGTH, GAME_LENGTH, WIN_REWARD, LOSE_REWARD, GREEN_REWARD, YELLOW_REWARD, GREY_REWARD

from ..strategies import random_strategy, fresh_letters_strategy, highest_ll_strategy, highest_ll_smart_strategy

class QWordle(gym.Env):
    """
    Class to implement a gym environment for Wordle.
    """

    def __init__(self):
        """
        Initialize the environment.
        """
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.MultiDiscrete(np.asarray([5,5,5]))

        self.strategies = {
            # 'random': random_strategy.RandomStrategy(),
            'fresh_letters': fresh_letters_strategy.FreshLettersStrategy(),
            'highest_ll': highest_ll_strategy.HighestLLStrategy(),
            'highest_ll_smart': highest_ll_smart_strategy.HighestLLSmartStrategy()
        }

    def reset(self):
        """
        Reset the environment.
        """
        self.guesses = np.full((GAME_LENGTH, WORD_LENGTH), -1)
        self.solution_word = random.choice(secret_words.secret_words)
        self.solution = word_to_action(self.solution_word)
        self.board = np.full((GAME_LENGTH, WORD_LENGTH), -1)
        self.letters = np.full((26,), -1)
        return np.asarray([0, 0, 0])

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
        try:
            strategy = self.strategies[action]
        except:
            strategy = list(self.strategies.values())[action]
        word = strategy.get_action({'board': self.board, 'guesses': self.guesses, 'letters': self.letters})
        word = word_to_action(word)
        res = self._check_guess(self.solution, word)
        game_row = np.where(self.board == -1)[0][0]
        self.board[game_row] = res

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
        if(np.all(self.board[game_row] == 2)):
            reward = WIN_REWARD
            done = True
            solved = True
        elif(game_row == GAME_LENGTH - 1):
            reward = LOSE_REWARD
            done = True
        else:
            reward = res.count(2)*GREEN_REWARD + res.count(1)*YELLOW_REWARD - res.count(0)*GREY_REWARD

        self.guesses[game_row] = word

        updated_state = np.asarray(list(get_state(self.letters).values())+[game_row])

        return(updated_state, reward, done, {'solved': solved})

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
    
