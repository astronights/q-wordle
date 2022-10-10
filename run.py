from src.models import no_learn_model

def main():
    model = no_learn_model.NoLearnModel()
    model.train()
    model.test()

if __name__ == '__main__':
    main()