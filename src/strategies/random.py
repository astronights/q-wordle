from src.strategies.base_strategy import BaseStrategy

import string
import numpy as np

from .base_strategy import BaseStrategy
from ..utils import action_to_word, word_to_action
from ..data import valid_words, secret_words
import random

class RandomStrategy(BaseStrategy):

    def __init__(self, config = None):
        super().__init__(config)
        self.words = np.setdiff1d(valid_words.words, secret_words.secret_words)

    def get_action(self, observations = None):
        letters = string.ascii_lowercase
        return ''.join(random.choices(letters,k=5))