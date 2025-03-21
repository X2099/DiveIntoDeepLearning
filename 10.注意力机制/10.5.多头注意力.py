# -*- coding: utf-8 -*-
"""
@File    : 10.5.多头注意力.py
@Time    : 2025/3/21 16:19
@Desc    : 
"""
import torch

import d2l

num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(key_size=num_hiddens, query_size=num_hiddens, value_size=num_hiddens,
                                   num_hiddens=num_hiddens, num_heads=num_heads, dropout=0.5)
attention.eval()

print(attention)

batch_size, num_queries = 2, 4  # num_queries: 查询的个数
num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))  # queries
print(X.shape)
# torch.Size([2, 4, 100]) (batch_size, num_queries, num_hiddens)

Y = torch.ones((batch_size, num_kvpairs, num_hiddens))  # keys, values
print(Y.shape)
# torch.Size([2, 6, 100]) (batch_size, num_kvpairs, num_hiddens)

print(attention(X, Y, Y, valid_lens).shape)
# torch.Size([2, 4, 100]) (batch_size, num_queries, num_hiddens)
