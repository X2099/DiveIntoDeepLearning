# -*- coding: utf-8 -*-
"""
@File    : 6.4.多输入多输出通道.py
@Time    : 2024/12/27 10:31
@Desc    : 
"""
import torch

from d2l import corr2d


def corr2d_multi_in(X, K):
    """
    对每个通道执行互相关操作，并将结果相加
    """
    Y = [corr2d(x, k) for x, k in zip(X, K)]
    # print("Y ---------------------- ")
    # print(Y)
    # """
    # Y ----------------------
    # [tensor([[19., 25.],
    #         [37., 43.]]), tensor([[37., 47.],
    #         [67., 77.]])]
    # """
    return sum(Y)


# 构造示例输入和卷积核
# X = torch.tensor([
#     [
#         [0.0, 1.0, 2.0],
#         [3.0, 4.0, 5.0],
#         [6.0, 7.0, 8.0]
#     ],
#     [
#         [1.0, 2.0, 3.0],
#         [4.0, 5.0, 6.0],
#         [7.0, 8.0, 9.0]
#     ]
# ])
# print(X.shape)  # torch.Size([2, 3, 3])
#
# K = torch.tensor([
#     [
#         [0.0, 1.0],
#         [2.0, 3.0]
#     ],
#     [
#         [1.0, 2.0],
#         [3.0, 4.0]
#     ]
# ])


# print(K.shape)  # torch.Size([2, 2, 2])
#
# # 执行卷积操作
# result = corr2d_multi_in(X, K)
# print(result.shape, '\n', result)
# """
# torch.Size([2, 2])
#  tensor([[ 56.,  72.],
#         [104., 120.]])
# """


def corr2d_multi_in_out(X, K):
    # 对每个输出通道执行互相关操作，并将所有结果堆叠在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], dim=0)


# 构造卷积核K，增加一个输出通道
# K = torch.stack((K, K + 1, K + 2), dim=0)
# print(K.shape)  # torch.Size([3, 2, 2, 2])

# 执行卷积操作
# result = corr2d_multi_in_out(X, K)
# print(result.shape)  # torch.Size([3, 2, 2])
# print(result)

def corr2d_multi_in_out_1x1(X, K):
    """
    该函数通过将输入 X 和卷积核 K 转换为矩阵形式，然后使用矩阵乘法进行运算，模拟了 1x1 卷积操作。
    对于每个像素位置，1x1 卷积核相当于对所有输入通道的加权求和，生成对应的输出通道。
    """
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))  # 展平输入
    K = K.reshape((c_o, c_i))  # 展平卷积核
    Y = torch.matmul(K, X)  # 全连接层中的矩阵乘法
    return Y.reshape((c_o, h, w))  # 恢复形状


# 使用一些样本数据验证
X = torch.normal(0, 1, size=(3, 3, 3))
K = torch.normal(0, 1, size=(2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
print(Y1.shape)  # torch.Size([2, 3, 3])
Y2 = corr2d_multi_in_out(X, K)
print(Y2.shape)  # torch.Size([2, 3, 3])
assert float(torch.abs(Y1 - Y2).sum()) < 10e-7
