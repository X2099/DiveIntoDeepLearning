# -*- coding: utf-8 -*-
"""
@File    : 4.4.模型选择、欠拟合和过拟合.py
@Time    : 2024/11/21 10:55
@Desc    : 
"""
import math
from pprint import pprint

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import d2l

# 设置参数
max_degree = 20  # 多项式最大阶数
n_train, n_test = 100, 100  # 训练和测试集大小
true_w = np.zeros(max_degree)  # 初始化系数
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])  # 只保留前四项的系数

# 生成特征
# 从标准正态分布中生成 (n_train + n_test, 1) 形状的随机数作为特征。
features = np.random.normal(size=(n_train + n_test, 1))
# 随机打乱生成的特征，以保证训练和测试数据的随机性
np.random.shuffle(features)

# 构造多项式特征矩阵
# 通过将每个特征升幂，生成多项式的各阶特征。poly_features 的形状为 (n_train + n_test, max_degree)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
print(poly_features.shape)  # (200, 20)
# 对每一列（对应不同阶数）进行归一化，除以阶数的阶乘（gamma(n) = (n-1)!）
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!

# 生成标签
# 根据多项式方程生成对应的标签值，即对 poly_features 和 true_w 的内积
labels = np.dot(poly_features, true_w)
# 在生成的标签上添加高斯噪声，模拟真实数据中的误差。噪声的标准差为 0.1
labels += np.random.normal(scale=0.1, size=labels.shape)
print(labels.shape)  # (200,)

# 转换为张量
# 将 numpy 数据转换为 PyTorch 张量，方便后续使用深度学习框架进行建模和训练
# 数据类型为 float32（单精度浮点数）
true_w, features, poly_features, labels = [
    torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]
]

print(poly_features[:n_train, :4].shape, poly_features[n_train:, :4].shape)
# 输出：torch.Size([100, 4]) torch.Size([100, 4])

print(labels[:n_train].shape, labels[n_train:].shape)
# 输出：torch.Size([100]) torch.Size([100])

print(poly_features[:10, :4])
"""
tensor([[ 1.0000e+00,  2.0456e-01,  2.0922e-02,  1.4266e-03],
        [ 1.0000e+00, -3.5547e-01,  6.3179e-02, -7.4861e-03],
        [ 1.0000e+00,  2.8585e-01,  4.0855e-02,  3.8928e-03],
        [ 1.0000e+00, -4.3815e-01,  9.5988e-02, -1.4019e-02],
        [ 1.0000e+00, -1.1253e+00,  6.3312e-01, -2.3748e-01],
        [ 1.0000e+00, -1.4531e+00,  1.0558e+00, -5.1140e-01],
        [ 1.0000e+00,  5.3094e-01,  1.4095e-01,  2.4945e-02],
        [ 1.0000e+00,  1.2345e+00,  7.6202e-01,  3.1357e-01],
        [ 1.0000e+00,  3.1758e-01,  5.0430e-02,  5.3386e-03],
        [ 1.0000e+00,  3.4239e-02,  5.8615e-04,  6.6896e-06]])
"""

print(labels[:10].shape)  # 输出：torch.Size([10])
print(labels[:10].reshape(-1, 1).shape)  # torch.Size([10, 1])


def evaluate_loss(net, data_iter, loss):
    """评估模型在数据集上的损失"""
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = DataLoader(TensorDataset(train_features, train_labels.reshape(-1, 1)), batch_size=batch_size,
                            shuffle=True)
    test_iter = DataLoader(TensorDataset(test_features, test_labels.reshape(-1, 1)), batch_size=batch_size,
                           shuffle=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            train_loss = evaluate_loss(net, train_iter, loss)
            test_loss = evaluate_loss(net, test_iter, loss)
            print(f"Epoch {epoch + 1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")
    print("Learned weights:", net[0].weight.data.numpy())
    print('Real weights:', true_w[:4])


# train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])

# train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])

train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])
