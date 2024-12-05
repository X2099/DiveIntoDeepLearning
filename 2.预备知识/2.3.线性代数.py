# -*- coding: utf-8 -*-
"""
@File    : 2.3.线性代数.py
@Time    : 2024/10/7 17:00
@Desc    : 
"""

import torch

A = torch.ones([4, 9])
print(torch.norm(A))  # tensor(6.)

# u = torch.tensor([3.0, -4.0])
# print(torch.abs(u).sum())  # tensor(7.)
#
# print(torch.norm(u))  # tensor(5.)

# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# print(A)
# """
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [12., 13., 14., 15.],
#         [16., 17., 18., 19.]])
# """
# B = torch.ones(4, 3)
# print(B)
# """
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]])
# """
# print(torch.mm(A, B))
"""
tensor([[ 6.,  6.,  6.],
        [22., 22., 22.],
        [38., 38., 38.],
        [54., 54., 54.],
        [70., 70., 70.]])
"""
# x = torch.arange(4, dtype=torch.float32)
# print(x, x.shape)  # tensor([0., 1., 2., 3.]) torch.Size([4])
#
# print(torch.mv(A, x))
# tensor([ 14.,  38.,  62.,  86., 110.])

# y = torch.ones(4, dtype=torch.float32)
# print(y)  # tensor([1., 1., 1., 1.])
# print(torch.dot(x, y))  # tensor(6.)
#
# print(x * y)  # tensor([0., 1., 2., 3.])
# print(torch.sum(x * y))  # tensor(6.)

# sum_A = A.sum(dim=1, keepdim=True)
# print(sum_A)
"""
tensor([[ 6.],
        [22.],
        [38.],
        [54.],
        [70.]])
"""
# print(sum_A.shape)  # torch.Size([5, 1])

# print(A / sum_A)
"""
tensor([[0.0000, 0.1667, 0.3333, 0.5000],
        [0.1818, 0.2273, 0.2727, 0.3182],
        [0.2105, 0.2368, 0.2632, 0.2895],
        [0.2222, 0.2407, 0.2593, 0.2778],
        [0.2286, 0.2429, 0.2571, 0.2714]])
"""

"""
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.],
        [16., 17., 18., 19.]])
"""
# print(A.cumsum(dim=0))
"""
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  6.,  8., 10.],
        [12., 15., 18., 21.],
        [24., 28., 32., 36.],
        [40., 45., 50., 55.]])
"""

# print(A.mean(dim=0))  # tensor([ 8.,  9., 10., 11.])
# print(A.sum(dim=0) / A.shape[0])  # tensor([ 8.,  9., 10., 11.])

# print(A.mean())  # tensor(9.5000)
# print(A.sum() / A.numel())  # tensor(9.5000)

# A_sum_axis0 = A.sum(dim=0)
# print(A_sum_axis0, A_sum_axis0.shape)
# tensor([40, 45, 50, 55]) torch.Size([4])

# A_sum_axis1 = A.sum(dim=1)
# print(A_sum_axis1, A_sum_axis1.shape)
# tensor([ 6, 22, 38, 54, 70]) torch.Size([5])

# print(A.sum(dim=[0, 1]))  # tensor(190)

# print(A.shape)  # torch.Size([5, 4])
# print(A.sum())  # tensor(190)

# x = torch.arange(4, dtype=torch.float32)
# print(x)  # tensor([0., 1., 2., 3.])
# print(x.sum())  # tensor(6.)

# a = 2
# X = torch.arange(24).reshape(2, 3, 4)
# print(X)
# """
# tensor([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]],
#
#         [[12, 13, 14, 15],
#          [16, 17, 18, 19],
#          [20, 21, 22, 23]]])
# """
# print(a + X)
# """
# tensor([[[ 2,  3,  4,  5],
#          [ 6,  7,  8,  9],
#          [10, 11, 12, 13]],
#
#         [[14, 15, 16, 17],
#          [18, 19, 20, 21],
#          [22, 23, 24, 25]]])
# """
# print(a * X)
# """
# tensor([[[ 0,  2,  4,  6],
#          [ 8, 10, 12, 14],
#          [16, 18, 20, 22]],
#
#         [[24, 26, 28, 30],
#          [32, 34, 36, 38],
#          [40, 42, 44, 46]]])
# """
# print((a * X).shape)  # 输出：torch.Size([2, 3, 4])

# 创建一个5x4的矩阵A，并克隆A到B
# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# B = A.clone()  # 复制A到B
#
# # 对于矩阵 𝐴 和 𝐵，它们的 Hadamard 积计算如下：
# print(A * B)
"""
tensor([[  0.,   1.,   4.,   9.],
        [ 16.,  25.,  36.,  49.],
        [ 64.,  81., 100., 121.],
        [144., 169., 196., 225.],
        [256., 289., 324., 361.]])
"""

# print(A)
"""
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.],
        [16., 17., 18., 19.]])
"""
# print(A + B)  # 矩阵A和B相加
"""
tensor([[ 0.,  2.,  4.,  6.],
        [ 8., 10., 12., 14.],
        [16., 18., 20., 22.],
        [24., 26., 28., 30.],
        [32., 34., 36., 38.]])
"""

# # 取反（Negation）
# x = torch.tensor([1.0, -2.0, 3.0])
# negated_x = -x
# print(negated_x)  # 输出：tensor([-1.,  2., -3.])
#
# # 指数（Exponential）
# x = torch.tensor([0.0, 1.0, 2.0])
# exp_x = torch.exp(x)
# print(exp_x)  # 输出：tensor([1.0000, 2.7183, 7.3891])
#
# # 取绝对值（Absolute Value）
# x = torch.tensor([-1.0, -2.0, 3.0])
# abs_x = torch.abs(x)
# print(abs_x)  # 输出：tensor([1., 2., 3.])

# # 创建一个形状为 (2, 3, 4) 的张量
# X = torch.arange(24).reshape(2, 3, 4)
#
# print(X)
# """
# tensor([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]],
#
#         [[12, 13, 14, 15],
#          [16, 17, 18, 19],
#          [20, 21, 22, 23]]])
# """

# B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
# print(B)
# """
# tensor([[1, 2, 3],
#         [2, 0, 4],
#         [3, 4, 5]])
# """
# print(B.T)
# """
# tensor([[1, 2, 3],
#         [2, 0, 4],
#         [3, 4, 5]])
# """
# print(B == B.T)
# """
# tensor([[True, True, True],
#         [True, True, True],
#         [True, True, True]])
# """
#
# # 创建一个5行4列的矩阵
# A = torch.arange(20).reshape(5, 4)
# print(A)
# """
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11],
#         [12, 13, 14, 15],
#         [16, 17, 18, 19]])
# """
#
# element = A[1, 2]
# print(element)  # 输出：tensor(6)
#
# print(A.T)
# """
# tensor([[ 0,  4,  8, 12, 16],
#         [ 1,  5,  9, 13, 17],
#         [ 2,  6, 10, 14, 18],
#         [ 3,  7, 11, 15, 19]])
# """

# x = torch.arange(4)  # 创建一个包含0到3的向量
# print(x)  # 输出: tensor([0, 1, 2, 3])
#
# element = x[3]
# print(element)  # 输出：tensor(3)
#
# length = len(x)  # 获取向量长度
# print(length)  # 输出: 4
#
# shape = x.shape
# print(shape)  # 输出: torch.Size([4])


# import torch
#
# # x = torch.tensor(3.0)  # 创建标量3.0
# y = torch.tensor(2.0)  # 创建标量2.0
#
# # 执行算术运算
# result_add = x + y  # 加法
# result_mul = x * y  # 乘法
# result_div = x / y  # 除法
# result_pow = x ** y  # 指数
#
# print(result_add, result_mul, result_div, result_pow)
#
# """
# tensor(5.) tensor(6.) tensor(1.5000) tensor(9.)
# """
