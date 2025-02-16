import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import *

"""
backend:nccl,mpi,gloo,ucc
"""
def init_process(rank:int, size:int, fn:Callable, backend='gloo'):
    """
    为每个进程初始化分布式环境，保证相互之间可以通信，并调用函数fn。

    torch.distributed.init_process_group()，该方法负责各进程之间的初始协调，保证各进程都会与master进行握手。该方法在调用完成之前会一直阻塞，并且后续的所有操作都必须在该操作之后。调用该方法时需要初始化下面的4个环境变量：
    MASTER_PORT：rank 0进程所在机器上的空闲端口；
    MASTER_ADDR：rank 0进程所在机器上的IP地址；
    WORLD_SIZE：进程总数；
    RANK：每个进程的RANK，所以每个进程知道其是否是master；
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size,init_method='env://')
    fn(rank, size)

    
def run(world_size:int, func:Callable):
    """
    启动world_size个进程，并执行函数func。

    函数run会根据传入的参数world_size，生成对应数量的进程。每个进程都会调用init_process来初始化分布式环境，并调用传入的分布式示例函数。
    """
    processes = []
    mp.set_start_method("spawn") # spawn:
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, func))
        p.start()
        processes.append(p)

    for p in processes:
        p.join() # 主进程等待所有进程

def p2p_block_func(rank, size):
    """
    一个简单的点对点通信实现。
    p2p_block_func实现从rank 0发送一个tensor([1.0])至rank 1，该操作在发送完成/接收完成之前都会阻塞。

    将rank src上的tensor发送至rank dst(阻塞)。
    """
    src = 0
    dst = 1
    group = dist.new_group(list(range(size)))
    # 对于rank src，该tensor用于发送
    # 对于rank dst，该tensor用于接收
    #tensor = torch.zeros(1).to(torch.device("cpu", rank))
    tensor = torch.zeros(1).to(torch.device("cpu", rank)) # 初始化为全0
    if rank == src:
        tensor += 1
        # 发送tensor([1.])
        # group指定了该操作所见进程的范围，默认情况下是整个world
        dist.send(tensor=tensor, dst=1, group=group)# 同步发送
    elif rank == dst:
        # rank dst的tensor初始化为tensor([0.])，但接收后为tensor([1.])
        print(f'Rank:{rank} before receive, data:{tensor}')
        # 从src那里接受tensor
        dist.recv(tensor=tensor, src=0, group=group) # 同步接收
    print(f'Rank:{rank} data:{tensor}')

def p2p_unblock_func(rank, size):
    """
    p2p_unblock_func是非阻塞版本的点对点通信。使用非阻塞方法时，因为不知道数据何时送达，所以在req.wait()完成之前不要对发送/接收的tensor进行任何操作。

    将rank src上的tensor发送至rank dst(非阻塞)。
    """
    src = 0
    dst = 1
    group = dist.new_group(list(range(size)))
    tensor = torch.zeros(1).to(torch.device("cpu", rank))
    if rank == src:
        tensor += 1
        # 非阻塞发送
        req = dist.isend(tensor=tensor, dst=dst, group=group) # Send a tensor asynchronously
        print(f"Rank:{rank} started sending")
    elif rank == dst:
        # 非阻塞接收
        req = dist.irecv(tensor=tensor, src=src, group=group)
        print(f"Rank:{rank} started receiving")

    req.wait() # 同步等待
    print(f'Rank:{rank} data:{tensor}')
    

def broadcast_func(rank, size):
    """
    broadcast_func会将rank 0上的tensor([1.])广播至所有的rank上。
    """
    src = 0
    group = dist.new_group(list(range(size)))
    if rank == src:
        # 对于rank src，初始化tensor([1.])
        tensor = torch.zeros(1).to(torch.device("cpu", rank)) + 1
    else:
        # 对于非rank src，初始化tensor([0.])
        tensor = torch.zeros(1).to(torch.device("cpu", rank))
    # 对于rank src，broadcast是发送；否则，则是接收
    dist.broadcast(tensor=tensor, src=0, group=group)
    print(f'Rank:{rank} data:{tensor}')

def reduce_func(rank, size):
    dst = 1
    group = dist.new_group(list(range(size)))
    tensor = torch.ones(1).to(torch.device("cpu", rank))
    # 对于所有rank都会发送, 但仅有dst会接收求和的结果
    dist.reduce(tensor, dst=dst, op=dist.ReduceOp.SUM, group=group)
    print(f'Rank:{rank} data:{tensor}')

def allreduce_func(rank, size):
    group = dist.new_group(list(range(size)))
    tensor = torch.ones(1).to(torch.device("cpu", rank))
    # tensor即用来发送，也用来接收
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print(f'Rank:{rank} data:{tensor}')

def gather_func(rank, size):
    dst = 1
    group = dist.new_group(list(range(size)))
    # 该tensor用于发送
    tensor = torch.zeros(1).to(torch.device("cpu", rank)) + rank
    gather_list = []
    if rank == dst:
        # gather_list中的tensor数量应该是size个，用于接收其他rank发送来的tensor
        gather_list = [torch.zeros(1).to(torch.device("cpu", dst)) for _ in range(size)]
        # 仅在rank dst上需要指定gather_list, 只有dst上收集tensor
        dist.gather(tensor, gather_list=gather_list, dst=dst, group=group)
        print(f'Rank:{rank} gather data:{gather_list}')
    else:
        # 非dst，相当于发送tensor
        dist.gather(tensor, dst=dst, group=group)
    print(f'Rank:{rank} data:{tensor}')

def allgather_func(rank, size):
    group = dist.new_group(list(range(size)))
    # 该tensor用于发送
    tensor = torch.zeros(1).to(torch.device("cpu", rank)) + rank
    # gether_list用于接收各个rank发送来的tensor
    gather_list = [torch.zeros(1).to(torch.device("cpu", rank)) for _ in range(size)]
    print(f'Rank:{rank} before gather, data:{tensor}')
    dist.all_gather(gather_list, tensor, group=group)
    # 各个rank的gather_list均一致
    print(f'Rank:{rank} data:{gather_list}')

def scatter_func(rank, size):
    src = 0
    group = dist.new_group(list(range(size)))
    # 各个rank用于接收的tensor
    tensor = torch.empty(1).to(torch.device("cpu", rank))
    if rank == src:
        # 在rank src上，将tensor_list中的tensor分发至不同的rank上
        # tensor_list：[tensor([1.]), tensor([2.])]
        tensor_list = [torch.tensor([i + 1], dtype=torch.float32).to(torch.device("cpu", rank)) for i in range(size)]
        # 将tensor_list发送至各个rank
        # 接收src发送的tensor中属于自己rank的那部分tensor, 比如第1个rank接收第1个数据，第2个rank接收第2个数据
        dist.scatter(tensor, scatter_list=tensor_list, src=0, group=group)
        print(f'Rank:{rank} tensor_list:{tensor_list}') 
    else:
        # 接收属于对应rank的tensor
        dist.scatter(tensor, scatter_list=[], src=0, group=group)
    # 每个rank都拥有tensor_list中的一部分tensor
    print(f'Rank:{rank} data:{tensor}') 

def reduce_scatter_func(rank, size):
    """


    假设有 4 个进程（world_size=4），每个进程的输入张量如下：
    Rank 0: [1, 2, 3, 4]

    Rank 1: [2, 3, 4, 5]

    Rank 2: [3, 4, 5, 6]

    Rank 3: [4, 5, 6, 7]

    执行 reduce_scatter 后，每个进程的输出张量为：

    Rank 0: [10]（1 + 2 + 3 + 4）

    Rank 1: [14]（2 + 3 + 4 + 5）

    Rank 2: [18]（3 + 4 + 5 + 6）

    Rank 3: [22]（4 + 5 + 6 + 7）

    输入张量：

    每个进程有一个输入张量 input_tensor，其内容根据 rank 不同而不同。

    如果使用 GPU，将张量移动到对应的 GPU 上。

    输出张量：

    output_tensor 用于存储 reduce_scatter 的结果。其大小是输入张量的一个分块。

    执行 reduce_scatter：

    dist.reduce_scatter(output_tensor, [input_tensor], op=dist.ReduceOp.SUM)：

    对所有进程的 input_tensor 执行规约操作（如求和）。

    将规约结果分散到每个进程的 output_tensor 中。

    # 销毁进程组
    dist.destroy_process_group()
    """
    group = dist.new_group(list(range(size)))
    # 用于接收的tensor
    tensor: torch.Tensor = torch.empty(1).to(torch.device("cpu", rank))
    # 用于发送的tensor列表
    # 对于每个rank，有tensor_list=[tensor([0.]), tensor([1.])]
    input_tensor_list = [torch.Tensor([1+i+rank]).to(torch.device("cpu", rank)) for i in range(size)]
    # step1. 经过reduce的操作会得到tensor列表[tensor([0.]), tensor([2.])]的求和
    # step2. tensor列表[tensor([0.]), tensor([2.])]分发至各个rank
    # rank 0得到tensor([0.])，rank 1得到tensor([2.])
    dist.reduce_scatter(tensor, input_tensor_list, op=dist.ReduceOp.SUM, group=group)
    print(f'Rank:{rank} data:{tensor} inputt_tensor:{input_tensor_list}') 

if __name__ == "__main__":
    #run(2, p2p_block_func) # 一个简单的同步阻塞式点对点通信实现。
    #run(2, p2p_unblock_func) # 异步非阻塞式点对点通信实现。 
    #run(3,broadcast_func) # broadcast_func会将rank 0上的tensor([1.])广播至所有的rank上。
    #run(3, reduce_func) # reduce_func会对group中所有rank的tensor进行聚合，并将结果发送至rank dst。
    #run(3, allreduce_func) # allreduce_func将group中所有rank的tensor进行聚合，并将结果发送至group中的所有rank。
    #run(3, gather_func) # gather_func从group中所有rank上收集tensor，并发送至rank dst。(相当于不进行聚合操作的reduce)
    #run(3, allgather_func) # allgather_func从group中所有rank上收集tensor，并将收集到的tensor发送至所有group中的rank。
    run(3, scatter_func) # scatter_func会将rank src中的一组tensor逐个分发至其他rank上，每个rank持有的tensor不同。
    #run(4, reduce_scatter_func) # 相当于每个进程只干一部分的事情，比如只在部分的数据上求和
