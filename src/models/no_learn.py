import gym
import numpy as np
from src.models.base_model import BaseModel
from src.utils import word_to_action
from ..envs.qwordle import QWordle

class NoLearn(BaseModel):
    def __init__(self, config=None):
        self.config = config
        self.env = QWordle()
        self.games_solved = []

    def train(self, iter = None):
        pass

    def test(self, verbose=True):
        obs = self.env.reset()
        done = False

        while not done:
            action = np.random.randint(0, 2)
            obs, reward, done, info = self.env.step(action)
            self.env.render()