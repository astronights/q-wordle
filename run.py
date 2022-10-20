from src.models import no_learn_model, qlearn

def main():
    # model = no_learn_model.NoLearnModel()
    model = qlearn.QLearn()
    model.train(1000)
    model.test()

if __name__ == '__main__':
    main()