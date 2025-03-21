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

"""
torchrun的程序一般具有如下结构：


def main(args):
	ddp_setup()				# 初始化进程池
	load_train_objs(args)	# 设置 dataset, model, optimizer, trainer 等组件，若存在 snapshot 则从中加载参数
	trian(args)				# 进行训练
	destroy_process_group()	# 销毁进程池

def train(args):
	for batch in iter(dataset):
		train_step(batch)
	
		if should_checkpoint:
			save_snapshot(snapshot_path)	# 用 rank0 保存 snapshot

if __name__ == "__main__":
	# 加载参数
    args = parser.parse_args()	
    
    # 现在 torchrun 负责在各个 GPU 上生成进程并执行，不再需要 mp.spawn 了
    main(args)
    

```shell
torchrun --standalone --nproc_per_node=ngpu test_torchrun.py

"""

def ddp_setup():
    # torchrun 会处理环境变量以及 rank & world_size 设置
    os.environ["MASTER_ADDR"] = "localhost"  # 由于这里是单机实验所以直接写 localhost
    os.environ["MASTER_PORT"] = "12350"  # 任意空闲端口
    init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    #init_process_group(backend="gloo", init_method='env://')

"""
 使用 DistributedDataParallel 进行单机多卡训练的基础上，使用 torchrun 进行容错处理，增强程序稳定性
 torchrun 允许我们在训练过程中按一定保存 snapshots，其中应当包含当前 epoch、模型参数（ckpt）、优化器参数、lr调度器参数等恢复训练所需的全部参数
 一旦程序出错退出，torchrun 会自动从最近 snapshots 重启所有进程
 除了增强稳定性外，torchrun 还会自动完成所有环境变量设置和进程分配工作，
 所以不再需要手动设置 local_rank 或用 mp.spawn 生成并分配进程
"""
class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            save_every: int,
            snapshot_path: str,  # 保存 snapshots 的位置
    ) -> None:
        self.gpu_id = int(os.environ['LOCAL_RANK'])  # torchrun 会自动设置这个环境变量指出当前进程的 rank, 即LOCAL_RANK
        self.origin_model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every  # 指定保存 snapshots 的周期
        self.epochs_run = 0  # 存储将要保存在 snapshots 中的 epoch num 信息
        self.snapshot_path = snapshot_path

        print(f"env:{os.environ}")

        # 若存在 snapshots 则加载，这样重复运行指令就能自动继续训练了
        if os.path.exists(snapshot_path):
            print('loading snapshot')
            self._load_snapshot(snapshot_path)

        # torchrun中冉要将模型用DDP包装一下
        self.parallel_model = DistributedDataParallel(self.origin_model, device_ids=[self.gpu_id])  # model 要用 DDP 包装一下

    def _load_snapshot(self, snapshot_path:str):
        ''' 加载 snapshot 并重启训练 '''
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.origin_model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad() # 清空梯度
        output = self.parallel_model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch:int):
        batch_size = len(next(iter(self.train_data))[0])
        if self.gpu_id == 0:
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {batch_size} | Total Steps: {len(self.train_data)}")

        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch:int):
        # 在 snapshot 中保存恢复训练所必须的参数
        snapshot_dict = {
            "MODEL_STATE": self.parallel_model.module.state_dict(),  # 由于多了一层 DDP 包装，通过 .module 获取原始参数
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot_dict, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        print(f"rank:{self.gpu_id} begin to train ...")
        for epoch in range(self.epochs_run, max_epochs):  # 现在从 self.epochs_run 开始训练，统一重启的情况
            self._run_epoch(epoch)

            # 各个 GPU 上都在跑一样的训练进程，这里指定 rank0 进程保存 snapshot 以免重复保存
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index:int):
        return self.data[index]

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(20, 40),
                                                   torch.nn.Linear(40, 1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = MyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,  # 设置了新的 sampler，参数 shuffle 要设置为 False
        sampler=DistributedSampler(dataset)  # 这个 sampler 自动将数据分块后送个各个 GPU，它能避免数据重叠
    )

def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    # 初始化进程池
    ddp_setup()

    # 进行训练
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)

    # 销毁进程池
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job with torchrun')
    parser.add_argument('--total-epochs', type=int, default=1000, help='Total epochs to train the model')
    parser.add_argument('--save-every', type=int, default=100, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=40, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    print(f"args:{args}")

    # 现在 torchrun 负责在各个 GPU 上生成进程并执行，不再需要 mp.spawn 了
    main(args.save_every, args.total_epochs, args.batch_size)

    '''
    运行命令: 
        torchrun --standalone  --nproc_per_node=2 test_torchrun.py # 单机多卡
    
    参数说明：
        --standalone 代表单机运行 
        --nproc_per_node=ngpu 代表使用所有可用GPU, 等于号后也可写gpu数量n, 这样会使用前n个GPU
    
    运行后获取参数：
        os.environ['RANK']          得到在所有机器所有进程中当前GPU的rank, 即global_rank
        os.environ['LOCAL_RANK']    得到在当前node中当前GPU的rank
        os.environ['WORLD_SIZE']    得到GPU的数量
    
    通过 CUDA_VISIBLE_DEVICES 指定程序可见的GPU, 从而实现指定GPU运行:
        CUDA_VISIBLE_DEVICES=0,3 torchrun --standalone --nproc_per_node=gpu multi_gpu_torchrun.py
    '''