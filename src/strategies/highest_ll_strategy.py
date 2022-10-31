from src.strategies.base_strategy import BaseStrategy


from .base_strategy import BaseStrategy
from ..data import word_ll, valid_words
import random
from ..utils import action_to_word, word_to_action

class HighestLLStrategy(BaseStrategy):

    def __init__(self, config = None):
        super().__init__(config)
        self.words = valid_words.words
        self.word_ll = word_ll.word_ll

    def get_action(self, observations = None):
        if(len(observations['guesses'])==0):
            return max(self.word_ll, key=self.word_ll.get)
        letters = set()
        for i, guess in enumerate(observations['guesses']):
            for j, color in enumerate(observations['board'][i]):
                if(color == 0):
                    letters.add(guess[j])

        filtered_words = [word for word in self.words if all(letter not in letters for letter in word_to_action(word))]
        filtered_word_ll = {word: self.word_ll[word] for word in filtered_words}
        if(len(filtered_word_ll) == 0):
            return max(self.word_ll, key=self.word_ll.get)
        return max(filtered_word_ll, key=filtered_word_ll.get)