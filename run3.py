from src.models import no_learn_model, qlearn, qlearn2, sarsa
import src.config as src_config

import pandas as pd
import numpy as np

from src.models import dqn, a2c

def main():
    dqn1 = dqn.DQNModel({'gamma': 0.9, 'alpha': 0.01, 'epsilon': 0.8})
    dqn1.train(10)
    # a2c1 = a2c.A2CModel({'gamma': 0.9, 'alpha': 0.01, 'epsilon': 0.8})
    # a2c1.train(100)
    # print(a2c1.games_solved)
    # a2c1.test(verbose=True)
            

if __name__ == '__main__':
    main()