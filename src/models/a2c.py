import gym
import sys
from src.models.base_model import BaseModel
from stable_baselines3.common.env_util import make_vec_env
from ..config import WIN_REWARD, LOSE_REWARD

import numpy as np
from stable_baselines3 import DQN, A2C

from ..envs.qwordle3 import QWordle3

class A2CModel(BaseModel):

    def __init__(self, config = None):
        super().__init__(config)
        self.epsilon = config['epsilon']
        self.gamma = config['gamma']
        self.alpha = config['alpha']
        self.games_solved = []

    def train(self, iter = 100):
        env = gym.make("QWordle3-v0")
        model = A2C("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=6*iter, log_interval=1000)
        try:
            model.learn(total_timesteps=6*iter, log_interval=1)
        except KeyboardInterrupt:
            pass
        rewards = model.replay_buffer.rewards

        # Get the index of the games solved:
        counter = 0
        for reward in rewards:
            if reward == 30:
                self.games_solved.append(counter)
            if reward == WIN_REWARD or reward == LOSE_REWARD:
                counter += 1
        model.save("wordle_a2c")
        return model

    def test(self, verbose=True):

        model = A2C.load("wordle_a2c")

        env = QWordle3()
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, res = env.step(action)
            if(verbose):
                env.render()
        if(verbose):
            print(res)
        return(res)