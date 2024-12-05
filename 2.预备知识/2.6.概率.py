# -*- coding: utf-8 -*-
"""
@File    : 2.6.概率.py
@Time    : 2024/10/28 17:04
@Desc    : 
"""
import torch
from torch.distributions import multinomial
import d2l

# 已知数据
p_disease = 0.01  # 患病概率
p_test_positive_given_disease = 0.95  # 阳性给定患病的概率
p_test_positive_given_no_disease = 0.05  # 阳性给定无病的概率

# 贝叶斯定理计算
p_positive = (p_test_positive_given_disease * p_disease) + (p_test_positive_given_no_disease * (1 - p_disease))
p_disease_given_positive = (p_test_positive_given_disease * p_disease) / p_positive

print(p_disease_given_positive)

# """
# 为了抽取一个样本，即掷骰子，我们只需传入一个概率向量。
# 输出是另一个相同长度的向量：它在索引 i 处的值是采样结果中出现的次数。
# """
# fair_probs = torch.ones([6]) / 6
# print(fair_probs)  # tensor([0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667])
# # 用于从多项分布（multinomial distribution）中采样1次
# result_1 = multinomial.Multinomial(1, fair_probs).sample()
# print(result_1)  # tensor([0., 0., 0., 0., 1., 0.])
#
# """
# 在估计一个骰子的公平性时，我们希望从同一分布中生成多个样本。
# 如果用Python的for循环来完成这个任务，速度会慢得惊人。
# 因此我们使用深度学习框架的函数同时抽取多个样本，得到我们想要的任意形状的独立样本数组。
# """
# result_10 = multinomial.Multinomial(10, fair_probs).sample()
# print(result_10)  # tensor([3., 2., 2., 3., 0., 0.])
#
# """
# 现在我们知道如何对骰子进行采样，我们可以模拟1000次投掷。
# 然后，我们可以统计1000次投掷后，每个数字被投中了多少次。
# 具体来说，我们计算相对频率，以作为真实概率的估计。
# """
# result_1000 = multinomial.Multinomial(1000, fair_probs).sample()
# print(result_1000)  # tensor([151., 163., 176., 190., 164., 156.])
# true_probs = result_1000 / 1000
# print(true_probs)  # tensor([0.1640, 0.1840, 0.1560, 0.1620, 0.1780, 0.1560])
#
# counts = multinomial.Multinomial(10, fair_probs).sample(sample_shape=torch.Size([500]))
# print(counts.shape)  # torch.Size([500, 6])
# print(counts)
# """
# tensor([[0., 0., 3., 2., 2., 3.],
#         [0., 4., 3., 1., 1., 1.],
#         [2., 0., 3., 2., 1., 2.],
#         ...,
#         [2., 2., 2., 1., 1., 2.],
#         [1., 1., 1., 4., 1., 2.],
#         [0., 2., 2., 2., 2., 2.]])
# """
# cum_counts = counts.cumsum(dim=0)
# print(cum_counts.shape)  # torch.Size([500, 6])
# print(cum_counts)
# """
# tensor([[  0.,   0.,   3.,   2.,   2.,   3.],
#         [  0.,   4.,   6.,   3.,   3.,   4.],
#         [  2.,   4.,   9.,   5.,   4.,   6.],
#         ...,
#         [838., 844., 856., 822., 841., 779.],
#         [839., 845., 857., 826., 842., 781.],
#         [839., 847., 859., 828., 844., 783.]])
# """
# cum_counts_sum_dim1 = cum_counts.sum(dim=1, keepdim=True)
# print(cum_counts_sum_dim1.shape)  # torch.Size([500, 1])
# estimates = cum_counts / cum_counts_sum_dim1
# print(estimates)
# """
# tensor([[0.1000, 0.2000, 0.1000, 0.1000, 0.4000, 0.1000],
#         [0.2000, 0.2000, 0.1500, 0.1500, 0.2000, 0.1000],
#         [0.2333, 0.2000, 0.1000, 0.1667, 0.1333, 0.1667],
#         ...,
#         [0.1661, 0.1673, 0.1588, 0.1604, 0.1827, 0.1647],
#         [0.1661, 0.1671, 0.1587, 0.1605, 0.1828, 0.1647],
#         [0.1660, 0.1670, 0.1588, 0.1604, 0.1830, 0.1648]])
# """
#
# d2l.set_figsize((6, 4.5))
# for i in range(6):
#     d2l.plt.plot(estimates[:, i].numpy(), label=f"P(die={i + 1})")  # 绘制二维线条图
#
# d2l.plt.axhline(y=1 / 6, color='black', linestyle='dashed')
# d2l.plt.gca().set_xlabel("实验组")
# d2l.plt.gca().set_ylabel("估计概率")
# d2l.plt.legend()  # 显示图例
# d2l.plt.show()
