# -*- coding: utf-8 -*-
"""
@File    : 3.2.线性回归的从零开始实现.py
@Time    : 2024/10/31 10:38
@Desc    : 
"""
import random

import torch
import d2l


def synthetic_data(w, b, num_samples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, size=[num_samples, len(w)])
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, size=y.shape)
    return X, y.reshape([-1, 1])


true_w = torch.tensor([2, -3.4])
true_b = 4.2

features, labels = synthetic_data(true_w, true_b, 1000)

# print(features)
"""
tensor([[ 2.3677e+00,  7.5134e-01],
        [-3.7171e-01,  1.5981e-01],
        [-7.3188e-01,  1.0761e+00],
        ...,
        [-4.5435e-01,  4.1389e-01],
        [-1.1956e+00, -2.6251e-01],
        [-9.7987e-04,  1.2006e+00]])
"""
# print(labels)
"""
tensor([[ 6.3847],
        [ 2.8922],
        [-0.9378],
        ...,
        [ 1.8783],
        [ 2.6986],
        [ 0.1151]])
"""


# d2l.set_figsize((7, 5))
# d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# d2l.plt.xlabel("Feature")
# d2l.plt.ylabel("Label")
# d2l.plt.show()


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


w = torch.normal(0, 0.01, size=[2], requires_grad=True)
b = torch.zeros(1, requires_grad=True)


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size  # 更新参数
            param.grad.zero_()  # 清空梯度


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
batch_size = 10

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = net(X, w, b)
        l = loss(y_hat, y)  # 计算小批量损失
        l.sum().backward()  # 反向传播计算梯度
        sgd([w, b], lr, batch_size)  # 更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

"""
epoch 1, loss 0.033658
epoch 2, loss 0.000111
epoch 3, loss 0.000048
"""
print(w.shape, true_w.shape)

print(f'训练学习到的w值: {w.detach().numpy()}，w的真实值：{true_w.numpy()}')
print(f'训练学习到的b值 {b.detach().item()}，b的真实值：{true_b}')
