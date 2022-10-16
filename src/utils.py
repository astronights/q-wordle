import json
from colorama import Fore

def get_words(filename):
    words = None
    with open(filename) as f:
        words = json.load(f)
    return words

def sort_words(filename):
    words = get_words(filename)
    words.sort()
    with open(filename, 'w') as f:
        json.dump(words, f)

def word_to_action(word):
    return [ord(c) - ord('a') for c in word]

def action_to_word(action):
    return [chr(ord('A') + c) for c in action]

def get_state(guess, word, state=None):
    state = {'green': 0, 'yellow': 0, 'grey': 0} if state is None else state
    for i, c in enumerate(guess):
        if c == word[i]:
            state['green'] += 1
        elif c in word:
            state['yellow'] += 1
        else:
            state['grey'] += 1
    return state