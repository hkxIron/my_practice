from typing import *

from torch.serialization import load
from model import *

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp


def quantize_statistics_estimate(model:Union[Net, NetBN], test_loader):
    """
    ⽤⼀部分训练数据来估计 min、max
    :param model:
    :param test_loader:
    :return:
    """

    for i, (data, target) in enumerate(test_loader, start=1):
        output = model.quantize_forward(data)
        if i % 500 == 0:
            break
    print('direct quantization finish')


def full_inference(model:Union[Net, NetBN], test_loader):
    """
    全精度推理
    :param model:
    :param test_loader:
    :return:
    """
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Full Model Accuracy: {:.0f}%\n'.format(100. * correct / len(test_loader.dataset)))


def quantize_inference(model:Union[Net, NetBN], test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Quant Model Accuracy: {:.0f}%\n'.format(100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    batch_size = 64
    using_bn = True
    load_quant_model_file = None
    # load_model_file = None

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/', train=True, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    if using_bn:
        model = NetBN()
        model.load_state_dict(torch.load('ckpt/mnist_cnnbn.pt', map_location='cpu'))
        save_file = "ckpt/mnist_cnnbn_ptq.pt"
    else:
        model = Net()
        model.load_state_dict(torch.load('ckpt/mnist_cnn.pt', map_location='cpu'))
        save_file = "ckpt/mnist_cnn_ptq.pt"

    print("量化之前")
    for param in model.fc.parameters():
        print(param)

    model.eval()
    full_inference(model, test_loader)

    # 8bit量化
    num_bits = 8
    model.quantize(num_bits=num_bits)
    model.eval()
    print('Quantization bit: %d' % num_bits)


    if load_quant_model_file is not None:
        model.load_state_dict(torch.load(load_quant_model_file))
        print("Successfully load quantized model %s" % load_quant_model_file)
    
    # 接下来就是⽤⼀些训练数据来估计 min、max
    quantize_statistics_estimate(model, train_loader)

    # ----------  ----------
    torch.save(model.state_dict(), save_file)
    model.freeze()

    print("量化之后")
    for param in model.fc.parameters():
        print(param)

    for param in model.qfc.parameters():
        print(param)

    # 测试是否设备转移是否正确
    # model.cuda()
    # print(model.qconv1.M.device)
    # model.cpu()
    # print(model.qconv1.M.device)
    quantize_inference(model, test_loader)
