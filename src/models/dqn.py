import gym
import sys

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

from ..envs.qwordle3 import QWordle3

def train_model():
    env = gym.make("QWordle3-v0")
    model = DQN(
        "MlpPolicy", 
        env,
        gamma=0.99, 
        learning_rate=5e-4,
        learning_starts=10000,
        buffer_size=10000,
        exploration_fraction=0.8,
        exploration_final_eps=0.5,
        target_update_interval=1000,
        train_freq=1,
        verbose=1,
    )
    try:
        model.learn(total_timesteps=600, log_interval=1)
    except KeyboardInterrupt:
        pass
    model.save("wordle_dqn")
    return model

train_model()

model = DQN.load("wordle_dqn")

print(model.ep_info_buffer)

env = QWordle3()
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()