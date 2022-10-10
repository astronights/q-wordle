from .base_model import BaseModel


class RF(BaseModel):
    
    def __init__(self, config):
        super().__init__(config)
        

    def train(self,trial, iter):
        pass

    def test(self):
        pass