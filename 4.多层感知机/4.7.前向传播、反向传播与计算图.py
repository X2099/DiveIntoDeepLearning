# -*- coding: utf-8 -*-
"""
@File    : 4.7.前向传播、反向传播与计算图.py
@Time    : 2024/12/3 16:22
@Desc    : 
"""
import numpy as np

# 初始化输入、权重和目标
x = np.array([1.0, 2.0])  # 输入向量
W1 = np.array([[0.5, 0.2], [0.3, 0.7]])  # 隐藏层权重
W2 = np.array([[0.6, 0.8]])  # 输出层权重
y_true = np.array([1.5])  # 目标值


# 激活函数（ReLU）
def relu(x):
    return np.maximum(0, x)


# 前向传播
h = np.dot(W1, x)  # 隐藏层中间变量
print(h)  # [0.9 1.7]
z = relu(h)  # 激活
print(z)  # [0.9 1.7]
y_pred = np.dot(W2, z)  # 输出层结果

# 损失计算
loss = 0.5 * np.sum((y_pred - y_true) ** 2)
print(f"预测值: {y_pred}, 损失: {loss}")


# 输出：预测值: [1.9], 损失: 0.08000000000000006


# 激活函数的导数
def relu_grad(x):
    return (x > 0).astype(float)


# 反向传播
grad_y_pred = y_pred - y_true  # 输出层梯度
grad_W2 = np.dot(grad_y_pred.reshape(-1, 1), z.reshape(1, -1))  # 输出层权重梯度

grad_z = np.dot(W2.T, grad_y_pred)  # 隐藏层输出的梯度
grad_h = grad_z * relu_grad(h)  # 隐藏层中间变量的梯度
grad_W1 = np.dot(grad_h.reshape(-1, 1), x.reshape(1, -1))  # 隐藏层权重梯度

print(f"隐藏层权重梯度: {grad_W1}")
"""
隐藏层权重梯度: [[0.24 0.48]
 [0.32 0.64]]
"""
print(f"输出层权重梯度: {grad_W2}")
"""
输出层权重梯度: [[0.36 0.68]]
"""
