# -*- coding: utf-8 -*-
"""
@File    : 3.6.从零开始实现softmax回归.py
@Time    : 2024/11/12 10:49
@Desc    : 
"""
import torch

import d2l

batch_size = 256  # 小批量大小
num_inputs = 784  # 输入数据维度
num_outputs = 10  # 输出数据维度

# 初始化权重w和偏置b
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


def train_epoch(net, train_iter, loss, updater):
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
        else:
            l.sum().backward()
            updater(batch_size)
        metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_softmax(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = d2l.evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        print(f'epoch {epoch + 1}, train loss {train_metrics[0]:.3f}, '
              f'train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}')


lr = 0.1


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)





if __name__ == '__main__':

    num_epochs = 10

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # 初始状态，train acc 和 test acc 都为 1/10，随机的
    metric = d2l.Accumulator(3)

    for X, y in train_iter:
        y_hat = net(X)
        l = cross_entropy(y_hat, y)
        metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())

    train_metrics = metric[0] / metric[2], metric[1] / metric[2]
    test_acc = d2l.evaluate_accuracy(net, test_iter)

    print(f'epoch {0}, train loss {train_metrics[0]:.3f}, '
          f'train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}')
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    animator.add(0, train_metrics + (test_acc,))
    # 训练模型
    train_softmax(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    # 可视化优化过程
    animator.show()
    # 预测
    predict_ch3(net, test_iter)
