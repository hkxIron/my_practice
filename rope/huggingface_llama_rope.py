from typing import *

import torch
from transformers import PretrainedConfig

from utils.util import *
from torch import nn

"""
下面的实现来源于HuggingFace的transformers中的Llama model中的rope实现

https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L204
"""
def rotate_half(x :torch.Tensor):
    """Rotates half the hidden dims of the input."""
    # x: [batch, num_head, seq_len, head_dim]
    x1 = x[..., : x.shape[-1] // 2] # 最后一维取前半部分
    x2 = x[..., x.shape[-1] // 2 :] # 最后一维取后半部分
    """
    x.shape: [batch, num_head, seq_len, head_dim]
    [batch=0, num_head=0, seq_len=0, head_dim= [ -x(dim/2+1),
                                                 -x(dim/2+2),
                                                 -x(dim/2+3),
                                                  ...,
                                                  -x(dim),
                                                  ----
                                                  x0,
                                                  x1,
                                                  x2,
                                                  ...,
                                                  x(dim/2)
                                                ]
    ]    
    """
    return torch.cat((-x2, x1), dim=-1) # 注意：这里将x2取反了


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    对query,key应用rope,因为它们要进行内积
    q: [batch, num_head, seq_len, head_dim]
    k: [batch, num_key_value_heads, seq_len, head_dim]

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    """
    cos: 在最后一维dim维
    [batch, seq_len, dim]
    示例数据： 
    [batch, seq_len=m, [
                        cos(m*theta0),
                        cos(m*theta1),
                        cos(m*theta2),
                        ...
                        cos(m*theta(head_dim//2-2)),
                        cos(m*theta(head_dim//2-1)),
                        ---------- 
                        cos(m*theta0),
                        cos(m*theta1),
                        cos(m*theta2),
                        ...
                        cos(m*theta(head_dim//2-2)),
                        cos(m*theta(head_dim//2-1))
                        ]]
    """
    # cos, sin: [batch, seq_len, head_dim]
    # =>   [batch, num_head=1, seq_len, head_dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    """
    q:
    [batch==0, num_head==0, seq_len==0, head_dim= [ 
                                                 q0,
                                                 q1,
                                                 q2,
                                                 ...,
                                                 q(dim/2)
                                                 ----------
                                                 q(dim/2+1),
                                                 q(dim/2+2),
                                                 q(dim/2+3),
                                                  ...,
                                                 q(dim-1)
                                                ]

    cos: [batch, num_head=1, seq_len, dim]
    示例数据： 
    [batch, num_head=1, seq_len==m,dim=[
                        cos(m*theta0),
                        cos(m*theta1),
                        cos(m*theta2),
                        ...
                        cos(m*theta(head_dim//2-2)),
                        cos(m*theta(head_dim//2-1)),
                        ---------- 注意：后半部分与前半部分相同
                        cos(m*theta0),
                        cos(m*theta1),
                        cos(m*theta2),
                        ...
                        cos(m*theta(head_dim//2-2)),
                        cos(m*theta(head_dim//2-1))
                        ]]
    ]    

    rotate_half(q): 
    [batch==0, num_head==0, seq_len==0, head_dim=[-q(dim/2+1),
                                                 -q(dim/2+2),
                                                 -q(dim/2+3),
                                                  ...,
                                                  -q(dim-1),
                                                  ----
                                                  q0,
                                                  q1,
                                                  q2,
                                                  ...,
                                                  q(dim/2)
                                                ]

    sin: [batch,num_head=1, seq_len, dim]
    示例数据： 
    [batch, num_head=1, seq_len==m,dim=[
                        sin(m*theta0),
                        sin(m*theta1),
                        sin(m*theta2),
                        ...
                        sin(m*theta(head_dim//2-2)),
                        sin(m*theta(head_dim//2-1)),
                        ---------- 注意：后半部分与前半部分相同
                        sin(m*theta0),
                        sin(m*theta1),
                        sin(m*theta2),
                        ...
                        sin(m*theta(head_dim//2-2)),
                        sin(m*theta(head_dim//2-1))
                        ]]

    由q_embed = (q * cos) + (rotate_half(q) * sin)可得如下矩阵，
    q_embed:
    [batch==0, num_head==0, seq_len==0, head_dim= [ 
                                                     q0*cos(m*theta0)-q(dim/2+1)*sin(m*theta0), # [q0, q(dim/2+1)]作为一个复向量，然后对此复向量旋转m*theta0角度
                                                     q1*cos(m*theta1)-q(dim/2+2)*sin(m*theta1),
                                                     q2*cos(m*theta2)-q(dim/2+3)*sin(m*theta2),
                                                     ...,
                                                     q(dim/2)*cos(m*theta(dim/2))-q(dim-1)*sin(m*theta(dim/2)),
                                                     ----------
                                                     q(dim/2+1)*cos(m*theta0)+q0*sin(m*theta0), # [q0, q(dim/2+1)]作为一个复向量，旋转角度theta0
                                                     q(dim/2+2)*cos(m*theta1)+q1*sin(m*theta1),
                                                     q(dim/2+3)*cos(m*theta2)+q2*sin(m*theta2),
                                                     ...,
                                                     q(dim-1)*cos(m*theta(dim/2))+q(dim/2)*sin(m*theta(dim/2))
                                                    ]

    注意：这里q_embed与RoFormer中的RoPE构造公式不同, 即RoFormer中是将相邻位置(q0,q1)作为复数的实部与虚部，而huggingface中llama中是将(qi,q(i+d/2))分别作为复数的实部与虚部,
    meta实现的llama则与原RoFormer中的Rope保持一致

    hf transformers中的llama是构造cos,sin矩阵与q相乘，实现时是将[qi,q(i+dim/2)]作为复数的虚部与实部
    facebook中的llama是构造复数e(i*m*theta),直接与q相乘，实现时与本文ROPE一致，是将相邻元素[qi,q(i+1)]作为复数的虚部与实部
    但无论哪种方式，虽然他们实现复数的方式不同，但他们的结果是一样的,他们相乘后均为q^T*R(n-m)*k，而R(n-m)只与n-m相关，因此即使训练与预测时的rope实现方式不同，但内积之后的结果是相同的，因此无需适配。

    theta值为: [batch, dim/2, 1] = 1 / 10000 ^ (even_dim_index / dim)

    RoFormer中的Rope
    = [q0, q1, q2, ..., q(d-2), q(d-1)] .* [cos(m*theta0), cos(m*theta0), cos(m*theta1),cos(m*theta1), ..., cos(m*theta(head_dim/2)),cos(m*theta(dim/2))]  .*代表逐元素相乘
    + [-q1,q0,-q3, ...,-q(d-1), q(d-2)] .* [sin(m*theta0), sin(m*theta0), sin(m*theta1),sin(m*theta1), ..., sin(m*theta(head_dim/2)),sin(m*theta(dim/2))]
    = [
       q0*cos(m*theta0)-q1*sin(m*theta0), # 即[q0,q1]作为复向量,然后对此复向量旋转m*theta0角度
       q1*cos(m*theta0)+q0*sin(m*theta0), # 即[q0,q1]作为复向量

       q2*cos(m*theta1)-q3*sin(m*theta1),
       q3*cost(m*theta1)+q2*sin(m*theta1),
       ...
       q(d-2)*cos(m*theta(dim/2))-q(d-1)*sin(m*theta(dim/2)), # [q(d-2), q(d-1)]作为复向量
       q(d-1)*cos(m*theta(dim/2))-q(d-2)*sin(m*theta(dim/2))
    ]
    """
    # q: [batch, num_head, seq_len, head_dim]
    # k: [batch, num_key_value_heads, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def _compute_default_rope_parameters(
        config: Optional[PretrainedConfig] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
        **rope_kwargs,
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"] # 原始论文为：base=10000, 10000^(-2*index/head_dim)
        dim = rope_kwargs["dim"] # dim一般为MHA中的head_dim
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        dim = int((config.hidden_size // config.num_attention_heads) * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    # base=10000
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度
    # inv_freq shape: [dim / 2], 其值为: 1 / [ 10000 ^ (even_dim_index / dim) ]
    # 当 dim_index=0时，inv_freq=1
    # 当 dim_index=dim时，inv_freq=1/10000
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor


class HuggingFaceLlamaRotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim=None,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            scaling_factor=1.0,
            rope_type="default",
            config = None,
            #config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type  # llama的rope_type是'default'
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = _compute_default_rope_parameters

        # inv_freq shape: [dim//2], 其值为: 1 / 10000 ^ (even_dim_index / dim)
        # inv_freq = 1.0 / [base ** (range(0, dim, 2)/dim)], 为一个向量
        # inv_freq就是ROPE中的theta角度
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # buffer就是常量，不会计算梯度
        self.original_inv_freq = self.inv_freq  # 取的buffer值

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()  # 注意：这里是no_grad
    def forward(self, x, position_ids):
        # 其实此处只用了x的device_type

        # x: [batch_size, num_key_value_heads, seq_len, head_dim], x为query或key
        # position_ids: [batch_size, sequence_length]

        # if "dynamic" in self.rope_type:
        #     self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        # position_ids: [batch, sequence_length]
        # inv_freq shape: [dim/2], 其值为: 1 / 10000 ^ (even_dim_index / dim)
        # => [1, dim/2, 1],其中dim=head_dim
        # inv_freq_expand:[batch, dim/2, 1], expand只适用于复制维度为1的
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)

        # position_ids: [batch_size, sequence_length]
        # position_ids_expanded: [batch, 1, sequence_length]
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # inv_freq_expand:[batch, dim/2, 1], 其值为: 1 / 10000 ^ (even_dim_index / dim)
            # position_ids_expanded: [batch, 1, seq_len],
            # freqs: [batch, dim/2, seq_len]
            # 转置 => [batch, seq_len, dim/2]
            # 即公式里的：pos / (10000^(2*i/dim)),
            # position_embed(m) = e^(j*m*theta) = e^(j*m/[10000^(2i/dim)]),其中j为虚数单位
            """
            inv_freq @ position_ids的物理意义是对batch的每条样本都将inv_freq向量复制一份

            freqs: 在最后一维head_dim维
            [batch, seq_len, head_dim/2]
            freqs示例数据： 
            [batch, seq_len=m, [m*theta0,
                                m*theta1,
                                m*theta2,
                                ... 
                                m*theta(head_dim//2-2),
                                m*theta(head_dim//2-1)
                                ]]
            """
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # freqs: [batch, seq_len, dim/2]
            # emb:  [batch, seq_len, dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            """
            cos: 在最后一维dim维
            [batch, seq_len, dim]
            示例数据： 
            [batch, seq_len==m, [
                                cos(m*theta0),
                                cos(m*theta1),
                                cos(m*theta2),
                                ...
                                cos(m*theta(head_dim//2-2)),
                                cos(m*theta(head_dim//2-1)),
                                ---------- 
                                cos(m*theta0),
                                cos(m*theta1),
                                cos(m*theta2),
                                ...
                                cos(m*theta(head_dim//2-2)),
                                cos(m*theta(head_dim//2-1))
                                ]]

            sin: 在最后一维dim维
            [batch, seq_len==m, dim]
            示例数据： 
            [batch, seq_len=m, [
                                sin(m*theta0),
                                sin(m*theta1),
                                sin(m*theta2),
                                ...
                                sin(m*theta(head_dim//2-2)),
                                sin(m*theta(head_dim//2-1)),
                                ---------- 
                                sin(m*theta0),
                                sin(m*theta1),
                                sin(m*theta2),
                                ...
                                sin(m*theta(head_dim//2-2)),
                                sin(m*theta(head_dim//2-1)),
                                ]]
            """
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling  # attention_scaling默认为1
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
