from src.models import no_learn_model, qlearn, sarsa
import src.config as src_config

import pandas as pd
import numpy as np

def main():
    # model = no_learn_model.NoLearnModel()
    model = qlearn.QLearn({'epsilon': 0.8, 'gamma': 0.6, 'alpha': 0.5})
    # model = sarsa.SARSALearn({'epsilon': 0.8, 'gamma': 0.6, 'alpha': 0.5})
    model.train(100)
    model.test()
    print(f"Games solved: {model.games_solved}, # Solved: {len(model.games_solved)}")
    q_table = model.Q
    actions = pd.DataFrame(columns=['# Green', '# Yellow', 'Game State', 'Action'])
    print(f'Q values shape: {q_table.shape}') #Num green, Num yellow, game step, action
    q_actions = np.argmax(q_table, axis=3)
    for green in range(src_config.WORD_LENGTH):
        for yellow in range(src_config.WORD_LENGTH-green):
            for state in range(src_config.GAME_LENGTH):
                action = model.strategies[q_actions[green, yellow, state]].__class__.__name__[:-8]
                actions = actions.append({'# Green': green, '# Yellow': yellow, 'Game State': state, 'Action': action}, ignore_index=True)
    print(actions.pivot(index=['# Green', '# Yellow'], columns=['Game State'], values=['Action']))
            

if __name__ == '__main__':
    main()