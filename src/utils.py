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

color_map = {
        -1: Fore.BLACK,
        0: Fore.BLACK,
        1: Fore.YELLOW,
        2: Fore.GREEN,
    }