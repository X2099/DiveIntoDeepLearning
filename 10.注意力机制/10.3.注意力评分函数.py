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

X = torch.rand(2, 2, 4)  # 随机生成一个2x2x4的张量
valid_lens = torch.tensor([2, 3])  # 代表每个序列的有效长度

# 计算掩蔽softmax
masked_softmax_result = d2l.masked_softmax(X, valid_lens)
print(masked_softmax_result)

# 示例数据
queries = torch.normal(0, 1, (2, 1, 20))  # 批量大小为2，查询数为1，特征维度为20
keys = torch.ones((2, 10, 2))  # 键的数量为10，特征维度为2
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)  # 值的矩阵
valid_lens = torch.tensor([2, 6])  # 有效长度

# 创建加性注意力模型并计算注意力输出
attention = d2l.AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
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
