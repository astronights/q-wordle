import gym
import sys
from src.models.base_model import BaseModel
from ..config import WIN_REWARD, LOSE_REWARD

import numpy as np
from stable_baselines3 import DQN

from ..envs.qwordle import QWordle

class DQNLearn(BaseModel):

    def __init__(self, config = None):
        super().__init__(config)
        self.epsilon = config['epsilon']
        self.gamma = config['gamma']
        self.alpha = config['alpha']
        self.games_solved = []
        self.env = QWordle()

    def train(self, iter = 100):
        self.games_solved = []
        self.env.reset()
        model = DQN(
            "MlpPolicy", 
            self.env,
            gamma=self.gamma, 
            learning_rate=self.alpha,
            learning_starts=10000,
            buffer_size=6*iter,
            exploration_fraction=self.epsilon,
            exploration_final_eps=0.5,
            target_update_interval=1000,
            train_freq=1,
            verbose=1,
        )
        try:
            model.learn(total_timesteps=6*iter, log_interval=1)
        except KeyboardInterrupt:
            pass
        counter = 0
        rewards = model.replay_buffer.rewards
        for reward in rewards:
            if reward[0] == 30:
                self.games_solved.append(counter)
            if reward[0] == WIN_REWARD or reward[0] == LOSE_REWARD:
                counter += 1

        model.save("wordle_dqn")
        return model

    def test(self, verbose=True):

        model = DQN.load("wordle_dqn")

        obs = self.env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, res = self.env.step(action)
            if(verbose):
                self.env.render()
        if(verbose):
            print(res)
        return(res)