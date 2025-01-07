# -*- coding: utf-8 -*-
"""
@File    : 6.6.卷积神经网络（LeNet）.py
@Time    : 2024/12/31 15:43
@Desc    : 
"""
import torch
from torch import nn
import d2l

# 定义LeNet模型
net = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

# print(net)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(256)


# 检查数据形状
# for images, labels in train_iter:
#     print(f"Batch of images shape: {images.shape}")
#     print(f"Batch of labels shape: {labels.shape}")
#     break


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """
    使用GPU计算模型在数据集上的精度
    """
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            """
            这行代码的目的是确定模型的第一个参数（通常是权重）被存储在哪个设备上。
            通常这个信息是用于确定当前模型运行在 CPU 还是 GPU 上，
            在使用 GPU 进行训练时非常有用，特别是当模型和数据需要显式迁移到同一个设备上时。
            """
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            # 采用 均匀分布 来初始化权重
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel="epoch", xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'],
                            figsize=(9, 5), title="卷积神经网络（LeNet）")
    timer, num_batches = d2l.Timer(), len(train_iter)
    metric = d2l.Accumulator(3)
    train_l, train_acc, test_acc = 0, 0, 0
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    animator.show()


lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
