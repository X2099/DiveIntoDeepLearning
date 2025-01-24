# -*- coding: utf-8 -*-
"""
@File    : 7.4.含并行连结的网络（GoogLeNet）.py
@Time    : 2025/1/23 15:31
@Desc    : 
"""
import torch
from torch import nn
from torch.nn import functional as F

import d2l


# 定义Inception块
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        # c1--c4是每条路径的输出通道数
        super(Inception, self).__init__()
        # 路径1：1x1卷积
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 路径2：1x1卷积 + 3x3卷积
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 路径3：1x1卷积 + 5x5卷积
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 路径4：3x3最大池化 + 1x1卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(self.p2_1(x)))
        p3 = F.relu(self.p3_2(self.p3_1(x)))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


# 定义GoogLeNet模型
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        # 模块1：包含一个 7×7 卷积层和一个最大池化层，用于初步提取特征。
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 模块2：两个卷积层串联，分别是 1×1 和 3×3 卷积。
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 模块3：包含两个Inception块，每块的输出通道数逐步增加。
        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 模块4：五个Inception块堆叠，通道数分配更加复杂。
        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 模块5：两个Inception块后接全局平均池化层，最后连接全连接层输出分类结果。
        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        # 输出层
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return self.fc(x)


net = GoogLeNet(num_classes=10)
X = torch.randn(size=(1, 1, 96, 96))
for layer in [net.b1, net.b2, net.b3, net.b4, net.b5, net.fc]:
    X = layer(X)
    print(f"{layer.__class__.__name__} output shape:\t{X.shape}")

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(128, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
