import sys

#from .model import Model
from test_import.src.models.model import Model

class CNN(Model):
    def __init__(self):
        super().__init__()
        print("CNN init:", __name__)
    def forward(self, x):
        print(f"cnn forward, input:{x}")
        return x+1

