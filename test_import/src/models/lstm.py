import sys

#from .model import Model
from test_import.src.models.model import Model

class LSTM(Model):
    def __init__(self):
        super().__init__()
        print("LSTM init")

    def forward(self, x):
        print(f"lstm forward, input:{x}")
        return x+2
