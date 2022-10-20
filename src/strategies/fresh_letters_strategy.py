from src.strategies.base_strategy import BaseStrategy
from src.utils import action_to_word, word_to_action


from .base_strategy import BaseStrategy
from ..data import valid_words
import random

class FreshLettersStrategy(BaseStrategy):

    def __init__(self, config = None):
        super().__init__(config)
        self.words = valid_words.words

    def get_action(self, observations):
        if(len(observations['guesses'])==0):
            return random.choice(self.words)
        letters = set()
        for guess in observations['guesses']:
            for letter in guess:
                letters.add(letter)

        filtered_words = [word for word in self.words if all(letter not in letters for letter in word_to_action(word))]
        if(len(filtered_words) == 0):
            return ''.join(action_to_word(observations['guesses'][-1])).lower()
        return random.choice(filtered_words)