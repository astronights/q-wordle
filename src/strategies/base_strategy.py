from abc import abstractmethod

class BaseStrategy:
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_action(self, observations = None):
        raise NotImplementedError