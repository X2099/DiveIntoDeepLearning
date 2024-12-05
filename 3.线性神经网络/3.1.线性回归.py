# -*- coding: utf-8 -*-
"""
@File    : 3.1.线性回归.py
@Time    : 2024/10/30 15:56
@Desc    : 
"""
import torch

# 设置随机种子
torch.manual_seed(49)

# 生成输入数据 X 和真实权重、偏置
num_samples, num_features = 1000, 2

true_w = torch.tensor([2.0, -3.4])
true_b = torch.tensor(4.2)

X = torch.normal(0, 1, [num_samples, num_features])

y = X @ true_w + true_b
y += torch.normal(0, 0.01, y.shape)  # 添加噪声

# 初始化模型参数
w = torch.randn(num_features, requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 定义线性模型的正向传播
def linear_regression(X, w, b):
    return X @ w + b


# 定义均方误差损失函数
def mse_loss(y_pred, y):
    return ((y_pred - y) ** 2).mean() / 2


# 定义随机梯度下降优化算法
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 模型训练
def train(num_epochs, lr, batch_size):
    loss = None
    for epoch in range(num_epochs):
        # 将数据划分为小批量
        idx = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = idx[i:i + batch_size]
            x_batch, y_batch = X[batch_indices], y[batch_indices]
            # 前向传播计算损失
            y_pred = linear_regression(x_batch, w, b)
            loss = mse_loss(y_pred, y_batch)
            # 反向传播
            loss.backward()
            # 更新参数
            sgd([w, b], lr, batch_size)

        # 每隔一定周期输出当前损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')


# 训练模型
train(num_epochs=100, lr=0.03, batch_size=10)

print(f'误差 - 真实权重：{true_w.numpy()}, 学习到的权重：{w.detach().numpy()}')
print(f'误差 - 真实偏置：{true_b.item()}, 学习到的偏置：{b.item()}')
