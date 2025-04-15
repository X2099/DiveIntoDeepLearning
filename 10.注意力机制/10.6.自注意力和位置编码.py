# -*- coding: utf-8 -*-
"""
@File    : 10.6.自注意力和位置编码.py
@Time    : 2025/3/24 16:53
@Desc    : 
"""
import torch
import d2l

# num_hiddens, num_heads = 100, 5
# attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
#                                    num_hiddens, num_heads, 0.5)
# attention.eval()

# print(attention)
"""输出：

MultiHeadAttention(
  (attention): DotProductAttention(
    (dropout): Dropout(p=0.5, inplace=False)
  )
  (W_q): Linear(in_features=100, out_features=100, bias=False)
  (W_k): Linear(in_features=100, out_features=100, bias=False)
  (W_v): Linear(in_features=100, out_features=100, bias=False)
  (W_o): Linear(in_features=100, out_features=100, bias=False)
)
"""

# batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
# X = torch.ones((batch_size, num_queries, num_hiddens))

# print(attention(X, X, X, valid_lens).shape)
# 输出：torch.Size([2, 4, 100])


# encoding_dim, num_steps = 32, 60
# pos_encoding = d2l.PositionalEncoding(encoding_dim, 0)
# pos_encoding.eval()

max_len = 100
num_hiddens = 32
P = torch.zeros((1, max_len, num_hiddens))
print(P.shape)
X = torch.arange(max_len, dtype=torch.float32).reshape(
    -1, 1) / torch.pow(10000, torch.arange(
    0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
print(X)
print(X.shape)

# X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
# P = pos_encoding.P[:, :X.shape[1], :]
# d2l.plot(torch.arange(num_steps), P[0, :, :].T, xlabel='Row (position)',
#          figsize=(6.18, 3.82), legend=["Col %d" % d for d in torch.arange(6, 10)])

# for i in range(8):
#     print(f'{i}的二进制是：{i:>03b}')
#
# P = P[0, :, :].unsqueeze(0).unsqueeze(0)
# d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
#                   ylabel='Row (position)', figsize=(3.82, 6.18), cmap='Blues')
# d2l.plt.show()
