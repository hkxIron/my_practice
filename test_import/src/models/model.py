import sys
from test_import.src.utils.time_utils import show_time

class Model():
    def __init__(self):
        #print("Model init")
        show_time()

    def forward(self, x):
        print(f"Model forward, input:{x}")
