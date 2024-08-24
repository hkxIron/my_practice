import matplotlib.pyplot as plt
import numpy as np


def learning_rate(step:int, model_size:int, factor:float, warmup:int):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    scale=min(step ** (-0.5), step * warmup ** (-1.5))
    return factor * (model_size ** (-0.5) * scale)

def test_lr():
    x = np.arange(0, 4000)
    # 在前500步，一直是快速上升，后面逐渐减小
    f = np.frompyfunc(lambda a: learning_rate(a, model_size=512, factor=1.0, warmup=500), 1, 1)
    y = f(x)
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    test_lr()