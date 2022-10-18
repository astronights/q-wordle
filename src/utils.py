import json
import numpy as np

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

def get_state(letters):
    state = {'green': 0, 'yellow': 0, 'missing': 0} 
    for letter in letters:
        if letter == 0:
            state['missing'] += 1
        elif letter == 1:
            state['yellow'] += 1
        elif letter == 2:
            state['green'] += 1
    return state

def calculate_glp(words):
    count = len(words)
    prb = [[0 for _ in range(5)] for _ in range(26)]

    # get word counts
    for word in words:
        values = word_to_action(word)
        for idx, letter in enumerate(values):
            prb[letter][idx] += 1

    # get probabilities
    for letter in range(len(prb)):
        for pos in range(len(prb[letter])):
            #laplace
            if prb[letter][pos] == 0:
                prb[letter][pos] = 1/count
            else:
                prb[letter][pos] = prb[letter][pos]/count
    return prb