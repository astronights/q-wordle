import gym
import random
from . import envs

from .utils import get_words
from .config import WORDS_FILE

def main():
    env = gym.make('QWordle-v0')
    obs = env.reset()
    done = False

    words = get_words(WORDS_FILE)

    while not done:
        word = random.choice(words)
        action = [ord(c) - ord('a') for c in word]
        obs, reward, done, info = env.step(action)
        env.render()