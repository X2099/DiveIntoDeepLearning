# -*- coding: utf-8 -*-
"""
@File    : 7.6.残差网络（ResNet）.py
@Time    : 2025/1/26 9:57
@Desc    : 
"""
import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        # 初始化Residual块
        # input_channels: 输入的通道数
        # num_channels: 输出的通道数
        # use_1x1conv: 是否使用1x1卷积来调整输入的形状
        # strides: 卷积步幅，默认是1
        super().__init__()
        # 第一个卷积层，进行特征提取，保持输入大小
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        # 第二个卷积层，继续特征提取
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        # 如果需要调整输入形状，则使用1x1卷积
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        # 批量归一化层，帮助加速训练
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        """前向传播函数，定义数据如何通过该残差块流动"""

        # 经过第一个卷积层，批量归一化，ReLU激活
        Y = F.relu(self.bn1(self.conv1(X)))
        # 经过第二个卷积层和批量归一化
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        # 将输入与输出相加，实现残差连接
        Y += X
        return F.relu(Y)  # 最后通过ReLU激活函数


b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 卷积层
    nn.BatchNorm2d(64),  # 批量归一化
    nn.ReLU(),  # 激活函数
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化层
)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:  # 第一个块需要调整输入通道数
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))  # 之后的块不需要调整通道数
    return blk


b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))  # 第一个模块
b3 = nn.Sequential(*resnet_block(64, 128, 2))  # 第二个模块
b4 = nn.Sequential(*resnet_block(128, 256, 2))  # 第三个模块
b5 = nn.Sequential(*resnet_block(256, 512, 2))  # 第四个模块

net = nn.Sequential(
    b1, b2, b3, b4, b5,  # 前面几个模块
    nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化，输出一个1x1的特征图
    nn.Flatten(),  # 将特征图展平为一维向量
    nn.Linear(512, 10)  # 全连接层，输出10个类别
)

# X = torch.rand(size=(1, 1, 224, 224))
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape:\t', X.shape)

import d2l

lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
