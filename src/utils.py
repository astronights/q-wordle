import json
import numpy as np
import math

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
    state = {'green': 0, 'yellow': 0}
    for letter in letters:
        # if letter == 0:
        #     state['missing'] += 1
        if letter == 1:
            state['yellow'] += 1
        elif letter == 2:
            state['green'] += 1
        else:
            pass
    return state
'''
Description:
Compute the probabilities of each letter being green at a position

Parameters:
words: string[], the words for which to get loglikehood values 

Returns:
prb: float[][], the probability table
'''
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

'''
Description:
Compute the loglikelihood value of a given word being green

Parameters:
words: string[], the words for which to get loglikehood values
prb: float, probability table 

Returns:
a dictionary where each (key, value) pair is ("word": "loglikelihood_value")
'''
def calculate_LL(words, prb):
    return {word:sum([math.log(prb[letter][pos]) for pos, letter in enumerate(word_to_action(word))]) for word in words}

'''
Description:
Check if a given word can be used as candidate for smart selection

Parameters:
words: string, the word given
d: dict, dictionary where each (key, value) pair is ("letter": "letter_pos")

Returns:
boolean value indicating whether it is candidate or not
'''
def is_candidate(word, d):
    for letter, pos in d.items(): 
        if word[pos] != letter: return False
    return True 

'''
Description:
Retrieve the next word with highest loglikelihood without previous info

Parameters:
words: string[], the list of words for selection of the next word
prb: probability table for each letter being green at a position

Returns:
next word with highest loglikelihood
'''
def next_highest_LL(words, prb):
    lls = calculate_LL(words, prb)
    return max(lls, key=lls.get)

'''
Description:
Retrieve the next word with highest loglikelihood with previous info

Parameters:
words: string[], the list of words for selection of the next word
d: dict, dictionary where each (key, value) pair is ("letter": "letter_pos")
prb: probability table for each letter being green at a position

Returns:
next word with highest loglikelihood
'''
def next_highest_LL_smart(words, d, prb):
    lls = calculate_LL([word for word in words if is_candidate(word, d)], prb)
    return max(lls, key=lls.get)


def invert_dict(index):
    return { v: k for k, l in index.items() for v in l }