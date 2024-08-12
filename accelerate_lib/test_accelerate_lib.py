from accelerate import init_empty_weights
import torch.nn as nn
import sys


def test_init():
    with init_empty_weights():
        # 在处理大模型时， accelerate 库包含许多有用的工具。
        # init_empty_weights 方法特别有用，因为任何模型，无论大小，都可以在此方法的上下文 (context) 内进行初始化，而无需为模型权重分配任何内存。
        # 初始化过的模型将放在 PyTorch 的  meta 设备上，这是一种用于表征向量的形状和数据类型而无需实际的内存分配的超酷的底层机制。
        # model = nn.Sequential([nn.Linear(10000, 10000) for _ in range(1000)])  # This will take ~0 RAM!
        num = 10000
        model = nn.Linear(num, num)  # This will take ~0 RAM!

    print(model)


def test_sum():
    x = [["a", "b"], ["c", "d"]]
    s = sum(x, [])  # 将二维列表x进行展开
    print(s)
    s = sum(x, ["y", "z"])  # 将进行展开
    print(s)


if __name__ == "__main__":
    print("args:", sys.argv)
    # test_init()
    test_sum()
