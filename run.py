from src.models import no_learn_model, qlearn

def main():
    # model = no_learn_model.NoLearnModel()
    model = qlearn.QLearn({'epsilon': 0.8, 'gamma': 0.6})
    model.train(10)
    model.test()

if __name__ == '__main__':
    main()