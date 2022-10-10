from abc import abstractmethod


class BaseModel:

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def train(self, trial, iter):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        raise NotImplementedError