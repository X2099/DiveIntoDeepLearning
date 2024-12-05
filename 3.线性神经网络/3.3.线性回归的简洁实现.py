# -*- coding: utf-8 -*-
"""
@File    : 3.3.线性回归的简洁实现.py
@Time    : 2024/11/6 16:23
@Desc    : 
"""
import torch
from torch import nn
import d2l

num_examples = 1000  # 样本数
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, num_examples)

batch_size = 10
data_iter = d2l.load_array((features, labels), batch_size)

net = nn.Sequential(nn.Linear(2, 1))  # 定义线性模型

net[0].weight.data.normal_(0, 0.01)  # 初始化权重参数
net[0].bias.data.fill_(0)  # 初始化偏置参数

loss = nn.MSELoss()  # 均方差损失函数

# 小批量随机梯度下降（SGD）
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)  # 预测值与真实值计算损失
        trainer.zero_grad()  # 清除参数梯度
        l.backward(retain_graph=True)  # 方向传播
        trainer.step()  # 更新梯度

    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
