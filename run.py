from src.models import no_learn_model, qlearn, sarsa

def main():
    # model = no_learn_model.NoLearnModel()
    model = qlearn.QLearn({'epsilon': 0.8, 'gamma': 0.6, 'alpha': 0.5})
    model = sarsa.SARSALearn({'epsilon': 0.8, 'gamma': 0.6, 'alpha': 0.5})
    model.train(10)
    model.test()

if __name__ == '__main__':
    main()