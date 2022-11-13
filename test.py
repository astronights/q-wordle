from src.models import no_learn, qlearn, sarsa, dqn
import src.config as src_config

import pandas as pd
import numpy as np


def no_learn_model():
    model = no_learn.NoLearn()
    model.train(100)
    model.test(verbose=True)
    print(f"Games solved: {model.games_solved}, # Solved: {len(model.games_solved)}")

def qlearn_model():
    model = qlearn.QLearn({'epsilon': 0.8, 'gamma': 0.6, 'alpha': 0.5})
    model.train(10)
    model.test(verbose=True)
    print(f"Games solved: {model.games_solved}, # Solved: {len(model.games_solved)}")
    q_table = model.Q
    print(f'Q values shape: {q_table.shape}')

def sarsa_model():
    model = sarsa.SARSALearn({'epsilon': 0.8, 'gamma': 0.6, 'alpha': 0.5})
    model.train(10)
    model.test(verbose=True)
    print(f"Games solved: {model.games_solved}, # Solved: {len(model.games_solved)}")
    q_table = model.Q
    print(f'Q values shape: {q_table.shape}')

def dqn_model():
    model = dqn.DQNLearn({'epsilon': 0.8, 'gamma': 0.6, 'alpha': 0.5})
    model.train(10)
    model.test(verbose=True)
    print(f"Games solved: {model.games_solved}, # Solved: {len(model.games_solved)}")

def main():
    no_learn_model()
    qlearn_model()
    sarsa_model()
    dqn_model()
            

if __name__ == '__main__':
    main()