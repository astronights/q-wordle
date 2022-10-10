from src.strategies.base_strategy import BaseStrategy


from .base_strategy import BaseStrategy
from ..data import valid_words
import random

class RandomStrategy(BaseStrategy):

    def __init__(self, config = None):
        super().__init__(config)
        self.words = valid_words.words

    def get_action(self, observations = None):
        return random.choice(self.words)