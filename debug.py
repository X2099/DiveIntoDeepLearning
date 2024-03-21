# -*- coding: utf-8 -*-
"""
@File    : debug.py
@Time    : 2024/3/8 10:18
@Desc    : 
"""
import torch
from torch import nn


def corr2d(X, K):
    """
    计算二维的互相关运算
    :param X: 二维张量
    :param K: 卷积核
    :return: 卷积结果
    """
    xh, xw = X.shape
    kh, kw = K.shape
    Y = torch.zeros(size=(xh - kh + 1, xw - kw + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + kh, j:j + kw] * K).sum()
    return Y


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        """
        :param kernel_size: 卷积核形状
        """
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias


conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
