import torch
from utils.util import *

# blog: https://kexue.fm/archives/8265
def sin_cos_position_embedding(batch_size,
                               head_num,
                               max_seq_len,
                               head_dim):
    # position: (max_seq_len, 1)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(-1)
    # dim_index: [head_dim//2], 两两一组，分别代表复数实部和虚部
    dim_index = torch.arange(0, head_dim // 2, dtype=torch.float)  # 即公式里的i, i的范围是 [0,d/2]
    # theta(i) = 10000^(-2i/d), theta(0)=10000^0=1, theta(d/2)=10000^(-d/d)=1/10000
    theta = torch.pow(10000, -2 * dim_index / head_dim)

    # (max_seq_len, head_dim//2)
    # 即公式里的：pos / (10000^(2i/d)), position_embed(m) = e^(j*m*theta) = e^(j*m/[10000^(2i/d)]),其中j为虚数单位
    position_embeddings = position * theta

    # [max_seq_len, head_dim//2]
    # =>
    # [max_seq_len, head_dim//2, 2]
    position_embeddings = torch.stack([torch.sin(position_embeddings),
                                      torch.cos(position_embeddings)], dim=-1) # 在最后一维上stack,注意不是torch.cat

    # [batch, head_num, max_len, head_dim//2, 2]
    # torch中的repeat就是(整体复制)铺地砖样复制，比如[1,2,3] repeat(2)变成[1,2,3, 1,2,3]
    position_embeddings = position_embeddings.repeat((batch_size, head_num, *([1] * len(position_embeddings.shape))))  # 在batch,head_num两个维度重复，其他维度都是1不重复

    # [batch, head_num, max_len, head_dim//2, 2], 最后一维(sin, cos)
    # [batch, head_num, max_len, head_dim]
    # reshape后：偶数sin, 奇数cos了
    position_embeddings = torch.reshape(position_embeddings, (batch_size, head_num, max_seq_len, head_dim))
    """
    position_embeddings: 在最后一维head_dim维
    [batch, head_num, max_len, head_dim]
    示例数据： 
    [:, :, max_len=m, [ sin(m*theta0),
                        cos(m*theta0),
                        sin(m*theta1),
                        cos(m*theta1),
                        ...
                        sin(m*theta(head_dim//2)),
                        cos(m*theta(head_dim//2))
                        ]]
    """
    return position_embeddings

"""
对q,k分别在复数空间下进行m*theta旋转
这样可以使它们在attention时具有相对位置信息
"""
def RoPE(q:torch.Tensor, k:torch.Tensor):
    # q,k: (batch, head_num, max_len, head_dim)
    batch_size, head_num, max_len, head_dim = q.shape
    assert head_dim//2*2 == head_dim, "head_dim must be even"

    # pos_emb: [batch, head_num, max_len, head_dim]
    # 其值为： e^(j*m/[10000^(2i/head_dim)]), i为索引，2i为偶数, m为位置, 最后一维(sin, cos)
    """
    pos_emb:
    [:, :, max_len=m, [ sin(m*theta0), # 0
                        cos(m*theta0), # 1
                        sin(m*theta1), # 2
                        cos(m*theta1), # 3
                        ...
                        sin(m*theta(head_dim//2)),
                        cos(m*theta(head_dim//2))
                        ]]
    """
    pos_emb = sin_cos_position_embedding(batch_size, head_num, max_len, head_dim).to(q.device)

    # cos_pos,sin_pos: [batch, head, max_len, head_dim]
    # 看rope公式可知，相邻cos，sin之间是相同的，所以单元素复制一遍。如(1,2,3)变成(1,1,2,2,3,3)
    # pos_emb中head_dim维的偶数列是sin，奇数列是cos
    """
    sin_pos: 
    [:, :, max_len=m, [ sin(m*theta0), # 0
                        sin(m*theta0), # 1
                        sin(m*theta1), # 2
                        sin(m*theta1), # 3
                        ...
                        sin(m*theta(head_dim//2)),
                        sin(m*theta(head_dim//2))
                        ]]
    cos_pos: 
    [:, :, max_len=m, [ cos(m*theta0), # 0
                        cos(m*theta0), # 1
                        cos(m*theta1), # 2
                        cos(m*theta1), # 3
                        ...
                        cos(m*theta(head_dim//2)),
                        cos(m*theta(head_dim//2))
                        ]]
    
    """
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制
    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制

    # q,k: [batch,  head, max_len, head_dim]
    """
    q:[batch,  head, max_len, head_dim]
    数据如下:
    [:,:,max_len=m, [q0,
                     q1,
                     q2,
                     ...
                     q(head_dim)]
    ] 
    q_imag:
    [:,:,max_len=m, [-q1,
                     q0,
                     -q3,
                     q2,
                     -q5,
                     q4,
                     ...
                     -q(head_dim-1),
                     q(head_dim)]
    ] 
    """
    q_imag = torch.stack([-q[..., 1::2],  # 只取奇数位然后取负
                               q[..., ::2]],  # 只取偶数位
                         dim=-1)
    q_imag = q_imag.reshape(q.shape)  # reshape后就是正负交替了

    # 计算Rope(q)=q*pos(m, head_dim), *对应位置相乘
    """
    RoFormer中的Rope, 即RoFormer中是将相邻位置(q0,q1)作为复数的实部与虚部 , dim=head_dim
    = [q0, q1, q2, ..., q(d-2), q(d-1)] .* [cos(m*theta0), cos(m*theta0), cos(m*theta1),cos(m*theta1), ..., cos(m*theta(head_dim/2)),cos(m*theta(dim/2))]  .*代表逐元素相乘
    + [-q1,q0,-q3, ...,-q(d-1), q(d-2)] .* [sin(m*theta0), sin(m*theta0), sin(m*theta1),sin(m*theta1), ..., sin(m*theta(head_dim/2)),sin(m*theta(dim/2))]
    = [
       q0*cos(m*theta0)-q1*sin(m*theta0),
       q1*cost(m*theta0)+q0*sin(m*theta0),
       q2*cos(m*theta1)-q3*sin(m*theta1),
       q3*cost(m*theta1)+q2*sin(m*theta1),
       ...
      q(d-2)*cos(m*theta(dim/2))-q(d-1)*sin(m*theta(dim/2)),
      q(d-1)*cos(m*theta(dim/2))-q(d-2)*sin(m*theta(dim/2))
    ]
    
    """
    q = q * cos_pos + q_imag * sin_pos

    # 同上
    k_imag = torch.stack([-k[..., 1::2],
                                   k[..., ::2]], dim=-1)
    k_imag = k_imag.reshape(k.shape)
    # 计算Rope(k)=k*pos(m, head_dim), *对应位置相乘
    # q,k: (batch, head_num, max_len, head_dim)
    k = k * cos_pos + k_imag * sin_pos
    return q, k

if __name__ == "__main__":
    setup_seed(3407)

    batch_size, head_num, max_len, head_dim = 2, 3, 4, 6

    query = torch.randn([batch_size, head_num, max_len, head_dim])
    key = torch.randn_like(query)
    query_rope, key_rope = RoPE(query, key)
    print(query_rope.shape)
    print(key_rope.shape)
    print(query_rope)
