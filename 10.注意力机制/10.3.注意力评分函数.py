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

# X = torch.rand(2, 3, 6)  # 随机生成一个2x3x6的张量
# valid_lens = torch.tensor([3, 5])  # 代表每个序列的有效长度

# print(valid_lens.dim())
# valid_lens = torch.repeat_interleave(valid_lens, shape[1])
# print(valid_lens)
# X = X.reshape(-1, shape[-1])
# print(X)
# print(valid_lens.unsqueeze(-1))
# print(torch.arange(X.shape[-2]))
# print(valid_lens.unsqueeze(-1) > torch.arange(X.shape[-1]))


# 计算掩蔽softmax
# masked_softmax_result = d2l.masked_softmax(X, valid_lens)
# print(masked_softmax_result)

# 示例数据
queries = torch.normal(0, 1, (2, 1, 20))  # 批量大小为2，查询数为1，特征维度为20
keys = torch.ones((2, 10, 2))  # 键的数量为10，特征维度为2
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)  # 值的矩阵

valid_lens = torch.tensor([2, 6])  # 有效长度
#
# key_size = 2
# query_size = 20
# num_hiddens = 8
#
# W_k = nn.Linear(key_size, num_hiddens, bias=False)
# W_q = nn.Linear(query_size, num_hiddens, bias=False)
# w_v = nn.Linear(num_hiddens, 1, bias=False)
# queries, keys = W_q(queries), W_k(keys)
#
# print(queries.shape)
# print(queries.unsqueeze(2).shape)
# print(keys.unsqueeze(1).shape)
# features = queries.unsqueeze(2) + keys.unsqueeze(1)
# print(features)
#
# 创建加性注意力模型并计算注意力输出
attention = d2l.AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
attention.eval()
output = attention(queries, keys, values, valid_lens)

print(attention.attention_weights.shape, attention.attention_weights)











print(output)
#
#
#
#
#
# # 示例：查询和键的维度
# queries = torch.normal(0, 1, (2, 1, 2))  # 2 个查询，每个查询有 1 个词，特征维度是 2
# keys = torch.ones((2, 10, 2))  # 2 个样本，每个样本 10 个键，每个键的维度是 2
# values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)  # 10 个值，每个值的维度是 4
#
# valid_lens = torch.tensor([2, 6])  # 每个句子的有效长度
#
# attention = d2l.DotProductAttention(dropout=0.5)
# output = attention(queries, keys, values, valid_lens)
# print(output)
