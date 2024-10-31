from typing import List

import torch.utils.data
from torchvision import datasets
import os

def test_collate1():
    mnist = datasets.MNIST(root='data/', train=True)
    print(mnist[0])

    # collate_fn为lamda x:x时表示对传入进来的数据不做处理
    dataloader = torch.utils.data.DataLoader(dataset=mnist, batch_size=2, shuffle=True, collate_fn=lambda x:x)

    #batch: [(img1, label1), (img2, label2),...]
    for x in dataloader:
        print(x)
        break

def test_collate2():
    import matplotlib.pyplot as plt
    mnist = datasets.MNIST(root='data/', train=True)
    print(mnist[0])
    #plt.imshow(mnist[0][0])
    #plt.show()

    def collate(data):
        img = []
        label=[]
        for x in data:
            img.append(x[0])
            label.append(x[1])
        return img, label

    # collate_fn为lamda x:x时表示对传入进来的数据不做处理
    dataloader = torch.utils.data.DataLoader(dataset=mnist, batch_size=2, shuffle=True, collate_fn=collate)

    #batch: [(img1, img2), (label1, label2),...]
    for x in dataloader:
        print(x)
        break

def test_collate3():
    def collate(batch_list:List):
        assert type(batch_list) == list, f"must be list"
        batch_size = len(batch_list)
        # 得到图像的rgb值
        data = torch.cat([torch.Tensor(item[0].getdata()) for item in batch_list]).reshape(batch_size, -1)
        labels = torch.Tensor([item[1] for item in batch_list]).reshape(batch_size, -1)
        return data, labels

    import PIL.Image
    from torchvision import transforms
    import numpy as np
    print(os.path.abspath(os.curdir))
    mnist = datasets.MNIST(root='data/', train=True)
    print(mnist[0])
    #PIL.Image.Image()
    img = mnist[0][0]
    #print(np.array(img.getdata()))
    print(torch.Tensor(img.getdata()))
    #plt.imshow(mnist[0][0])
    #plt.show()

    # collate_fn为lamda x:x时表示对传入进来的数据不做处理
    dataloader = torch.utils.data.DataLoader(dataset=mnist,
                                             batch_size=5,
                                             shuffle=True,
                                             collate_fn=collate)

    #batch: [(img1, img2), (label1, label2),...]
    for idx, x in enumerate(dataloader):
        print(f"data {idx} batch, labels: {x}")
        if idx>=5:
            break

if __name__ == "__main__":
    #test_collate1()
    #test_collate2()
    test_collate3()
