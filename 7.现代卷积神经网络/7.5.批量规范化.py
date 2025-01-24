# -*- coding: utf-8 -*-
"""
@File    : 7.5.批量规范化.py
@Time    : 2025/1/24 14:21
@Desc    : 
"""
import torch
from torch import nn
import d2l


# 从头开始实现一个具有张量的批量规范化层
def batch_norm(X: torch.Tensor, gamma, beta, moving_mean, moving_var, eps, momentum):
    """
    批量归一化实现函数

    参数说明：
    X: 输入数据张量
       - 全连接层：形状为 [batch_size, num_features]
       - 卷积层：形状为 [batch_size, num_channels, height, width]
       - 说明：待归一化的输入数据。

    gamma: 缩放参数（可学习的）
       - 全连接层：形状为 [num_features]
       - 卷积层：形状为 [num_channels]
       - 说明：用于调整归一化后的数据的尺度，初始化时通常为 1。

    beta: 偏移参数（可学习的）
       - 全连接层：形状为 [num_features]
       - 卷积层：形状为 [num_channels]
       - 说明：用于调整归一化后的数据的偏移量，初始化时通常为 0。

    moving_mean: 全局均值（非可学习）
       - 全连接层：形状为 [num_features]
       - 卷积层：形状为 [num_channels]
       - 说明：训练时更新为当前批次均值的移动平均值，预测时直接使用。

    moving_var: 全局方差（非可学习）
       - 全连接层：形状为 [num_features]
       - 卷积层：形状为 [num_channels]
       - 说明：训练时更新为当前批次方差的移动平均值，预测时直接使用。

    eps: 防止分母为零的小常数
       - 类型：浮点数（float）
       - 说明：用于数值稳定性，通常设置为一个很小的值，例如 1e-5 或 1e-8。

    momentum: 动量参数
       - 类型：浮点数（float）
       - 说明：控制全局均值和方差的更新速度，常用值为 0.9 或 0.99。
    """
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)  # 均值
            var = ((X - mean) ** 2).mean(dim=0)  # 方差
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    # 对归一化后的数据进行缩放和偏移
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        # num_features：完全连接层的输出数量或卷积层的输出通道数。
        # num_dims：2表示完全连接层，4表示卷积层
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta,
                                                          self.moving_mean, self.moving_var,
                                                          eps=1e-5, momentum=0.9)
        return Y


# net = nn.Sequential(
#     nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
#     nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
#     nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
#     nn.Linear(84, 10)
# )

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16 * 4 * 4, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(256)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
