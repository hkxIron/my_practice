import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

# 对 python 多进程的一个 pytorch 包装
import torch.multiprocessing as mp

# 这个 sampler 可以把采样的数据分散到各个 CPU 上
from torch.utils.data.distributed import DistributedSampler

# 实现分布式数据并行的核心类
from torch.nn.parallel import DistributedDataParallel

# DDP 在每个 GPU 上运行一个进程，其中都有一套完全相同的 Trainer 副本（包括model和optimizer）
# 各个进程之间通过一个进程池进行通信，这两个方法来初始化和销毁进程池
from torch.distributed import init_process_group, destroy_process_group

from torch.distributed.fsdp import (
FullyShardedDataParallel,
CPUOffload,
)
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 4)
        self.layer2 = nn.Linear(4, 16)
        self.layer3 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return x

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(10), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index:int):
        return self.data[index]

# 初始化分布式环境
def setup_distributed():
    init_process_group(backend="nccl", init_method="env://")  # 使用 NCCL 后端（适用于 GPU）
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def train(save_every: int, total_epochs: int, batch_size: int):
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"local rank:{local_rank}")
    origin_model = MyModel().to(local_rank)
    # auto wrap fdsp
    fsdp_model = FullyShardedDataParallel(
        origin_model,
        #auto_wrap_policy=default_auto_wrap_policy, # 会根据层的参数量自动决定是否分片。你可以通过
        cpu_offload=CPUOffload(offload_params=True),
    )
    dataset = MyTrainDataset(10000)
    sampler = DistributedSampler(dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,  # 设置了新的 sampler，参数 shuffle 要设置为 False
        sampler=sampler
    )

    criterion = nn.CrossEntropyLoss() 
    optim = torch.optim.Adam(fsdp_model.parameters(), lr=0.0001)

    for epoch in range(total_epochs):
        print(f"epoch:{epoch}")
        for x, label in data_loader:
            x = x.to(local_rank)
            label = label.to(local_rank)
            optim.zero_grad()
            out = fsdp_model(x)
            loss = criterion(out, label)
            loss.backward()
            optim.step()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job with torchrun')
    parser.add_argument('--total-epochs', type=int, default=100, help='Total epochs to train the model')
    parser.add_argument('--save-every', type=int, default=10, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=40, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    print(f"args:{args}")

    setup_distributed()
    # 现在 torchrun 负责在各个 GPU 上生成进程并执行，不再需要 mp.spawn 了
    train(args.save_every, args.total_epochs, args.batch_size)
    destroy_process_group()
