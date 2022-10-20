from src.strategies.base_strategy import BaseStrategy


from .base_strategy import BaseStrategy
from ..data import word_ll
import random

class HighestLLStrategy(BaseStrategy):

    def __init__(self, config = None):
        super().__init__(config)
        self.word_ll = word_ll.word_ll

    def get_action(self, observations = None):
        return max(self.word_ll, key=self.word_ll.get)