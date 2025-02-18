# -*- coding: utf-8 -*-
"""
@File    : 10.3.注意力评分函数.py
@Time    : 2025/2/11 16:39
@Desc    : 
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

import d2l


def masked_softmax(X, valid_lens):
    """
    执行掩蔽 Softmax 操作。

    该函数会根据有效长度 `valid_lens` 对输入张量 `X` 进行掩蔽处理。掩蔽部分将被置为一个非常小的负数（例如 -1e6），
    以确保它们在 Softmax 操作中得到零的权重。

    参数：
    X (Tensor): 需要执行 Softmax 的输入张量。通常是一个 3D 张量，形状为 (batch_size, seq_len, num_classes)。
    valid_lens (Tensor): 有效长度张量。它指定了每个序列中有效的元素的个数。它的形状可以是一个 1D 张量，表示
                          每个序列的有效长度，或者是一个 2D 张量，每行表示对应样本中每个查询的有效长度。

    返回：
    Tensor: 执行掩蔽 Softmax 后的结果。与输入 `X` 形状相同，但掩蔽部分的权重会变成零。
    """
    if valid_lens is None:
        return F.softmax(X, dim=-1)

    # 获取 X 的形状 (batch_size, seq_len, num_classes)
    shape = X.shape
    if valid_lens.dim() == 1:
        # 如果 valid_lens 是一维的，将它扩展为二维，重复有效长度，适配每个序列
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        # 确保 valid_lens 是一个一维张量
        valid_lens = valid_lens.reshape(-1)
    # 将 X 展平，变成二维张量 (batch_size * seq_len, num_classes)
    X = X.reshape(-1, shape[-1])
    # 创建掩蔽操作，超出有效长度的位置被替换为一个非常小的负数（-1e6）
    X = torch.where(
        valid_lens.unsqueeze(-1) > torch.arange(X.shape[-1], device=X.device),
        X,  # 对应有效位置保留原值
        torch.tensor(-1e6)  # 对于无效位置，使用极小值 -1e6
    )
    # 执行 Softmax 操作，返回 Softmax 的结果，dim=-1 表示按最后一个维度（num_classes）计算 Softmax
    return F.softmax(X.reshape(shape), dim=-1)


X = torch.rand(2, 2, 4)  # 随机生成一个2x2x4的张量
valid_lens = torch.tensor([2, 3])  # 代表每个序列的有效长度

# 计算掩蔽softmax
masked_softmax_result = masked_softmax(X, valid_lens)
print(masked_softmax_result)


class AdditiveAttention(nn.Module):
    """加性注意力（Additive Attention）类"""

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        # 定义键（key）的线性变换层，将输入的键映射到隐藏层空间
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        # 定义查询（query）的线性变换层，将输入的查询映射到隐藏层空间
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        # 定义一个线性变换层，用于将注意力得分计算为一个标量
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        """
        前向传播方法，计算加性注意力的输出。

        参数：
        查询、键和值的形状为（批量大小，步数或词元序列长度，特征大小）
        queries: 查询张量，形状为 (batch_size, query_len, query_dim)
        keys: 键张量，形状为 (batch_size, key_len, key_dim)
        values: 值张量，形状为 (batch_size, key_len, value_dim)
        valid_lens: 有效长度，表示哪些键是有效的，通常用于掩蔽填充部分，形状为 (batch_size, query_len)

        返回值：
        注意力汇聚输出的形状为（批量大小，查询的步数，值的维度）
        返回加权后的值张量，形状为 (batch_size, query_len, value_dim)
        """
        # 将查询和键分别映射到隐藏空间
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 进行广播求和：查询和键的每一对进行加和，形状变为 (batch_size, query_len, key_len, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        # 通过tanh激活函数增加非线性变换
        features = torch.tanh(features)
        # 使用w_v计算加性注意力得分。输出形状为 (batch_size, query_len, key_len)
        scores = self.w_v(features).squeeze(-1)
        # 计算注意力权重，使用softmax进行归一化，注意masking掉无效位置
        self.attention_weights = masked_softmax(scores, valid_lens)
        # 使用注意力权重对值进行加权求和，得到最终的加性注意力输出
        # 使用 dropout 防止过拟合
        return torch.bmm(self.dropout(self.attention_weights), values)


# 示例数据
queries = torch.normal(0, 1, (2, 1, 20))  # 批量大小为2，查询数为1，特征维度为20
keys = torch.ones((2, 10, 2))  # 键的数量为10，特征维度为2
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)  # 值的矩阵
valid_lens = torch.tensor([2, 6])  # 有效长度

# 创建加性注意力模型并计算注意力输出
attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
attention.eval()
output = attention(queries, keys, values, valid_lens)


# print(output)


class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]  # 查询和键的维度

        # 计算查询和键的点积，注意进行缩放
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)

        # 计算注意力权重
        attention_weights = masked_softmax(scores, valid_lens)

        # 返回加权求和的值
        return torch.bmm(self.dropout(attention_weights), values)


# 示例：查询和键的维度
queries = torch.normal(0, 1, (2, 1, 2))  # 2 个查询，每个查询有 1 个词，特征维度是 2
keys = torch.ones((2, 10, 2))  # 2 个样本，每个样本 10 个键，每个键的维度是 2
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)  # 10 个值，每个值的维度是 4

valid_lens = torch.tensor([2, 6])  # 每个句子的有效长度

attention = DotProductAttention(dropout=0.5)
output = attention(queries, keys, values, valid_lens)
# print(output)
