# -*- coding: utf-8 -*-
"""
@File    : 6.2.图像卷积.py
@Time    : 2024/12/25 11:00
@Desc    : 
"""
import torch
from torch import nn


def corr2d(X, K):
    """
    互相关运算实现
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


class Conv2D(nn.Module):

    def __init__(self, kernel_size):
        """
        kernel_size: 卷积核形状
        """
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))  # 卷积核权重
        self.bias = nn.Parameter(torch.zeros(1))  # 卷积核偏置

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias


X = torch.ones((6, 8))
X[:, 2:6] = 0
# print(X)

K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
# print(Y)

conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), bias=False)

print("学习前的卷积核权重为：", conv2d.weight.data)

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

lr = 3e-2

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= lr * conv2d.weight.grad

    print(f'epoch {i + 1}, loss {l.sum():.3f}')

print("学习到的卷积核权重为：", conv2d.weight.data)

# X = torch.tensor([
#     [0.0, 1.0, 2.0],
#     [3.0, 4.0, 5.0],
#     [6.0, 7.0, 8.0]
# ])

# net = Conv2D((2, 2))
# print(net(X))
#
# K = torch.tensor([
#     [0.0, 1.0],
#     [2.0, 3.0]
# ])
#
# print(corr2d(X, K))
