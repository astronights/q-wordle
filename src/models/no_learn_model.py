import gym
from src.models.base_model import BaseModel
from src.utils import word_to_action
from .. import envs
from ..strategies.random import RandomStrategy
from ..strategies.highest_ll_strategy import HighestLLStrategy
from ..strategies.highest_ll_smart_strategy import HighestLLSmartStrategy
from ..strategies.fresh_letters_strategy import FreshLettersStrategy

class NoLearnModel(BaseModel):
    def __init__(self, config=None):
        self.config = config
        self.strategy = RandomStrategy()
        self.env = gym.make('QWordle-v0')

    def train(self, iter = None):
        pass

    def test(self):
        obs = self.env.reset()
        done = False

        while not done:
            word = self.strategy.get_action(obs)
            action = word_to_action(word)
            obs, reward, done, info = self.env.step(action)
            self.env.render()