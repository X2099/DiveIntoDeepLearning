# -*- coding: utf-8 -*-
"""
@File    : 7.2.使用块的网络（VGG）.py
@Time    : 2025/1/14 16:17
@Desc    : 
"""
import torch
from torch import nn
import d2l


def vgg_block(num_convs, in_channels, out_channels):
    """
    定义一个VGG块，由若干个卷积层和一个最大池化层组成。

    参数：
    - num_convs (int): 当前块中卷积层的数量。
    - in_channels (int): 输入特征图的通道数。
    - out_channels (int): 输出特征图的通道数。

    返回：
    - nn.Sequential: 包含所有卷积层和池化层的顺序容器。
    """
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    """
    定义VGG网络，由多个卷积块和全连接层组成。

    参数：
    - conv_arch (list): VGG块的配置列表，其中每个元素是一个元组，格式为
      (num_convs, out_channels)，表示卷积块中卷积层的数量和输出通道数。

    返回：
    - nn.Sequential: 包含卷积块和全连接层的顺序容器。
    """
    conv_blks = []
    in_channels = 1  # 假设输入为单通道灰度图（例如Fashion-MNIST）

    # 构造卷积块部分
    for num_convs, out_channels in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels  # 更新输入通道数为当前块的输出通道数

    # 构造全连接层部分
    return nn.Sequential(
        *conv_blks,  # 卷积块
        nn.Flatten(),
        nn.Linear(in_channels * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )


# 定义VGG-11的架构
conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
net = vgg(conv_arch)

# 测试VGG网络的输出形状
X = torch.randn(size=(1, 1, 224, 224))  # 假设输入为 224x224 的单通道图像
for blk in net:
    X = blk(X)
    print(f"{blk.__class__.__name__} output shape: {X.shape}")

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs = 0.05, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
