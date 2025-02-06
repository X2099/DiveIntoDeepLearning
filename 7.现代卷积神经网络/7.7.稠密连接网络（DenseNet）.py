# -*- coding: utf-8 -*-
"""
@File    : 7.7.稠密连接网络（DenseNet）.py
@Time    : 2025/1/26 14:35
@Desc    : 
"""
import torch
from torch import nn


def conv_block(input_channels, num_channels):
    # 定义卷积块：每个卷积块包含批量归一化、ReLU激活函数和卷积操作
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )


class DenseBlock(nn.Module):
    # 定义稠密块（DenseBlock）：由多个卷积块组成
    def __init__(self, num_convs, input_channels, num_channels):
        super().__init__()
        layer = []
        # 创建多个卷积块，并按顺序连接
        for i in range(num_convs):
            # 每个卷积块的输入通道数等于输入通道数 + 每个卷积块输出通道数的累计和
            layer.append(conv_block(input_channels + num_channels * i, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)  # 前向传播，获取当前卷积块的输出
            X = torch.cat((X, Y), dim=1)  # 在通道维度上连接输入X和输出Y，增加特征的通道数
        return X


def transition_block(input_channels, num_channels):
    # 定义过渡层（TransitionBlock）：用于减少特征图的通道数，并通过池化层减少空间维度（高、宽）
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),  # 1x1卷积层，减少通道数
        nn.AvgPool2d(kernel_size=2, stride=2)  # 使用2x2的平均池化，步幅为2，减半高和宽
    )


# 定义初始卷积层
# 输入通道数为1（灰度图像），输出分类为10个类别（例如Fashion-MNIST数据集）
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# 稠密块和过渡层的构建
num_channels, growth_rate = 64, 32  # 初始通道数为64，增长率为32
num_convs_in_dense_blocks = [4, 4, 4, 4]  # 每个稠密块包含4个卷积层
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    # 添加一个稠密块，使用当前通道数和增长率
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    num_channels += num_convs * growth_rate

    # 添加过渡层，在每个稠密块后面，除最后一个稠密块外，其他都需要减小通道数并进行池化
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))  # 通道数减半
        num_channels = num_channels // 2

# 将所有块连接成一个完整的网络
net = nn.Sequential(
    b1,  # 初始卷积层和池化层
    *blks,  # 连接所有的稠密块和过渡层
    nn.BatchNorm2d(num_channels),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化，将输出大小调整为1x1
    nn.Flatten(),
    nn.Linear(num_channels, 10)
)

# 打印网络结构
# print(net)

import d2l

lr, num_epochs, batch_size = 0.1, 1, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
