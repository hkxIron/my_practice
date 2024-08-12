import sys
import os
import time
from test_import.src.models.lstm import LSTM # 每层加了__init__.py这样不报错
# 用相对路径会报错
#from ..models.cnn import CNN # 报错 ImportError: attempted relative import with no known parent package
#from ...test_import.src.models.cnn import CNN # ImportError: attempted relative import with no known parent package

from test_import.src.models.cnn import CNN
from test_import.src.models.transformer import Transformer
from test_import.src.utils.compare_util import compare_model
from test_import.src.utils.path_utils import show_paths
from test_import.src.utils.time_utils import show_time

def test_compare():
    print("begin to test1")
    show_time()
    compare_model()

def test_cnn():
    print("begin to test2")
    cnn_model = CNN()
    print(cnn_model.forward(10))

if __name__ == "__main__":
    #print("args:", sys.argv)
    show_paths()
    test_compare()
    test_cnn()
