import sys

#from .model import Model
from test_import.src.models.model import Model

class Transformer(Model):
    def __init__(self):
        super().__init__()
        print("transformer init")

    def forward(self, x):
        print(f"transformer forward, input:{x}")
        return x+3
