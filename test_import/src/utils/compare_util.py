import sys
from test_import.src.models.lstm import LSTM # 每层加了__init__.py这样不报错
# 用相对路径会报错
#from ..models.cnn import CNN # 报错 ImportError: attempted relative import with no known parent package
#from ...test_import.src.models.cnn import CNN # ImportError: attempted relative import with no known parent package

from test_import.src.models.cnn import CNN
from test_import.src.models.transformer import Transformer

def compare_model():
    cnn_model = CNN()
    lstm_model = LSTM()
    transformer_model = Transformer()
    cnn_val = cnn_model.forward(1)
    lstm_val = lstm_model.forward(1)
    transformer_val = transformer_model.forward(1)
    print(f"eval model")
    print(f"cnn_val:{cnn_val}")
    print(f"lstm_val:{lstm_val}")
    print(f"transformer_val:{transformer_val}")


if __name__ == "__main__":
    print("args:", sys.argv)
    compare_model()
