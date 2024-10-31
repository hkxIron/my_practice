from typing import *

import torch
from transformers import PretrainedConfig

from utils.util import *
from torch import nn


"""
下面的实现来源于meta的llama3.1中的Llama model中的rope实现

https://github.com/meta-llama/llama-models/blob/2fe1a1690162910660332e3294a552cf0ec7e754/models/llama3/reference_impl/model.py#L95
"""

"""
precompute_freqs_cis中的cis是 "cosine" 和 "sine" 的缩写，它经常在数学中使用来表示复数的极坐标形式。
具体来说，给定一个角度theta，其对应的复数可以表示为：
cis(theta) = cos(theta) + i*sin(theta), 即一般形式的欧拉公式
"cis" 表示的是一个复数，其实部是角度θ的余弦值，而虚部是角度θ的正弦值, theta表示幅角 ,这种表示方法在复数分析、信号处理等领域中非常有用。

因此，故名思义，该函数的目的是预计算一个复数频率张量。该函数有两个入参，dim和end。
dim就是每个attention_head中的维度，在这里就是head_dim = hidden/head_num=4096/32=128。
end是self.params.max_seq_len * 2，也就是4096，这也是Llama2最大的token处理数量。计算过程解释见注释：
"""


def precompute_freqs_cis(
        dim: int,  # head_dim =128
        seq_length: int,  # max_seq_len*2
        theta: float = 10000.0,
        use_scaled: bool = False
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        seq_length (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    # dim = head_dim = 128
    # seq_len = max_seq_len*2 = 4096
    # 幅角最小单位：10000^(-2i/dim)
    # torch.arange(0, dim, 2) [0, 2, 4, 6, 8, 10,..., 124, 126] 共64个
    # torch.arange(0, dim, 2)[: (dim // 2)] 保证是64个
    # freqs = [1/10000.0^(0/128), 1/10000.0^(2/128), 1/10000.0^(4/128), ..., 1/10000.0^(126/128)]
    # freqs.shape: [dim//2]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # index_of_seq: [0, 1, 2, ..., 4095]
    # index_of_seq.shape:[seq_len]
    index_of_seq = torch.arange(seq_length, device=freqs.device, dtype=torch.float32)
    # if use_scaled:
    #     freqs = apply_scaling(freqs)

    """
    freqs 得到 freqs和t的笛卡尔积
    freqs:[seq_length, embed_dim//2] =（4096，64）
    freqs = [[0, 0, 0,..., 0],
             [1/10000.0^(0/128), 1/10000.0^(2/128), 1/10000.0^(4/128), ..., 1/10000.0^(126/128)],
             [2/10000.0^(0/128), 2/10000.0^(2/128), 2/10000.0^(4/128), ..., 2/10000.0^(126/128)],
             ...,
             [4095/10000.0^(0/128), 4095/10000.0^(2/128), 4095/10000.0^(4/128), ..., 4095/10000.0^(126/128)]]
    其公式值为：
    [
        0*theta(0), 0*theta(1), ..., 0*theta(dim/2-1),
        1*theta(0), 1*theta(1), ..., 1*theta(dim/2-1),
        ...
        m*theta(0), m*theta(1), ..., m*theta(dim/2-1),
        ...
        seq_len*theta(0), seq_len*theta(1), ..., seq_len*theta(dim/2-1),
        ]
    """
    # index_of_seq.shape:[seq_len]
    # freqs:[dim//2]
    # freqs_angle:[seq_len, dim//2]
    freqs_angle = torch.outer(index_of_seq, freqs)

    """
    在PyTorch中，torch.polar用于通过极坐标（magnitude和angle）来创建一个复数张量。
    这个函数接受两个张量作为输入：一个张量包含复数的模（magnitude，也就是复数的长度），
    另一个张量包含复数的角度（angle，也就是复数的相位角），然后返回一个相应的复数张量。
    下面就是创建模长为1的，有不同相位角的复数张量。
    freqs_cis:[seq_len, dim//2]
    """
    freqs_cis = torch.polar(abs=torch.ones_like(freqs), angle=freqs_angle)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    """
    注意freqs_cis的维度并不是（4096，64），而是截取了seqlen的一部分，freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]。
    """
    # freqs_cis.shape = [1024, 64]
    # x.shape = [2, 1024, 32, 64]
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # freqs_cis:[seq_length, embed_dim//2]
    # x:[batch, query_seqlen, head_num, head_dim/2]
    # 将freqs_cis.shape变为[batch=1, query_seqlen=1024, head_num=1, head_dim/2=64]
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


"""
与其它实现不同，meta的实现直接在复数空间相乘得到rope编码，即
f(q,m)=q_complex*e^(i*m*theta)
"""


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.



    """
    # xq:[batch, seqlen, head_num, head_dim]
    # xk:[batch, seqlen, n_kv_head, head_dim]

    """
    将xq和xk的最后一个维度进行复数运算，得到新的xq和xk
    为了进行复数运算，需要将xq和xk的最后一个维度展开为2维
    例如，xq的形状为[2, seq_len, 32, 128], reshape后为[2, seq_len, 32 , 64, 2]
    view_as_complex函数可以将张量中的最后一维的两个元素作为实部和虚部合成一个复数

    xq:[batch, query_seqlen, head_num, head_dim]
    -> [batch, query_seqlen, head_num, head_dim/2, 2]
    torch.view_as_complex:其中输入张量的最后一个维度必须为2, 将相邻位置(q0,q1)作为复数的实部与虚部,其中偶数部分为实部，奇数部分为虚部
    具体而言，其中的复数为：[x0+j*x(1), x2+j*x3, ...., x(dim_2)+j*x(dim-1)], 长度为head_dim/2
    此处reshape会将相邻位置当成复数的实部与虚部
    xq_complex: [batch, query_seqlen, head_num, head_dim/2]
    """
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))

    # xk:[batch, seqlen, n_kv_head, head_dim]
    # -> [batch, key_seqlen, head_num, head_dim/2, 2]
    # xk_complex: [batch, key_seqlen, head_num, head_dim/2]
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    """
    freqs_cis:[seq_length, embed_dim//2]
    其值为：
    [
        0*theta(0), 0*theta(1), ..., 0*theta(dim/2-1),
        1*theta(0), 1*theta(1), ..., 1*theta(dim/2-1),
        ...
        m*theta(0), m*theta(1), ..., m*theta(dim/2-1),
        ...
        seq_len*theta(0), seq_len*theta(1), ..., seq_len*theta(dim/2-1)
    ]

    xq_complex: [batch, query_seqlen, head_num, head_dim/2]

    将freqs_cis.shape变为[batch=1, query_seqlen=1024, head_num=1, head_dim/2=64]
    """
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)

    """
    ROPE编码, f(q, m) = q_complex*e^(i*m*theta)
    其具有相对位置信息：<f(q,m), f(k,n)> = g(q,k,m-n) = (q.T)*R(n-m)*k
    即将xq转为复数后，与位置m的复数相乘，得到rope

    xq_complex: [batch, query_seqlen, head_num, head_dim/2]
    freqs_cis:  [batch=1, query_seqlen=1024, head_num=1, head_dim/2=64]
    view_as_real和view_as_complex相反，可以将张量中最后一维的复数拆出实部和虚部

    xq_complex*freqs_cis为复数相乘, 即模长相乘，幅角相加,由于freqs_cis模长为1,因此只有幅角相加
    xq_rope_complex.shape:[batch, seqlen, head_num, head_dim/2],结果为复数，有实部与虚部
    xq_rope_real.shape = [batch=2, seq_len, head_num=32 , head_dim/2=64, 2]
    xq_rope: flatten(3)将张量展平为[batch=2, seq_len, head_num=32, head_dim=128]，3代表从的第3个维度开始展平, 即虚数的实部与虚部又分别作为矩阵的相邻元素

    xq_rope:[batch, query_seqlen=1024, head_num, head_dim]
    xk_rope:[batch, query_seqlen=1024, head_num, head_dim]
    """
    xq_rope_complex = xq_complex * freqs_cis
    xq_rope_real = torch.view_as_real(xq_rope_complex)  # 在复数空间中旋转后又还原成实数
    xq_rope = xq_rope_real.flatten(3)  # 从第3维head_dim//2开始，将后面所有维（head_dim//2, 2）展平

    xk_rope = torch.view_as_real(xk_complex * freqs_cis).flatten(3)

    """
    最终,xq_rope的复数表示为 
    [batch==0, seq_len==0, head_num==0, head_dim= [ 
                                                   q0*cos(0*theta0)-q1*sin(0*theta0),
                                                   q1*cos(1*theta0)+q0*sin(1*theta0),
                                                   q2*cos(2*theta1)-q3*sin(2*theta1),
                                                   q3*cos(3*theta1)+q2*sin(3*theta1),
                                                   ...
                                                   q3*cos(m*theta1)+q2*sin(m*theta1),
                                                   ...
                                                   q(d-2)*cos(m*theta(dim/2))-q(d-1)*sin(m*theta(dim/2)),
                                                   q(d-1)*cos(m*theta(dim/2))-q(d-2)*sin(m*theta(dim/2))
                                                ]
  ]
    """
    return xq_rope.type_as(xq), xk_rope.type_as(xk)
