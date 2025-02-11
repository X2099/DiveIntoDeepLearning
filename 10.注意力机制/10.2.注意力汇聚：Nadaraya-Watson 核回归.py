# -*- coding: utf-8 -*-
"""
@File    : 10.2.注意力汇聚：Nadaraya-Watson 核回归.py
@Time    : 2025/2/8 11:00
@Desc    : 
"""
import torch
from torch import nn
import d2l

n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)  # 排序后的训练样本
print(x_train)


def f(x):
    return 2 * torch.sin(x) + x ** 0.8


y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
print(y_train)

x_test = torch.arange(0, 5, 0.1)  # 测试样本
print(x_test)
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数
print(n_test)


def plot_kernel_reg(y_hat):
    """
    绘制 Nadaraya-Watson 核回归的预测结果与真实值的对比图。

    参数:
    y_hat (Tensor 或 ndarray): 预测值，与测试输入 x_test 对应。

    说明:
    - 该函数首先绘制测试数据点 (x_test) 的真实值 (y_truth) 与模型预测值 (y_hat)。
    - 然后，在图中以散点的形式标注训练数据点 (x_train, y_train)。
    - `d2l.plot` 用于绘制曲线，`d2l.plt.plot` 用于绘制散点。
    """
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y',
             legend=['Truth', 'Pred'], xlim=[0, 5], ylim=[-1, 5])
    # 'o' 是 matplotlib.pyplot.plot() 的 marker 参数，用于指定数据点的标记样式。
    # 具体来说，'o' 表示用圆形（circle）标记数据点。
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)


# y_hat = torch.repeat_interleave(y_train.mean(), n_test)

# X_repeat的形状:(n_test,n_train), 每一行都包含着相同的测试输入
# X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# print(X_repeat.shape)  # torch.Size([50, 50])
# # attention_weights的形状：(n_test,n_train), 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
# attention_weights = nn.functional.softmax(-(X_repeat - x_train) ** 2 / 2, dim=1)
# print(attention_weights.shape)  # torch.Size([50, 50])


# y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重


# 定义带参数的注意力汇聚模型
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # 定义一个可学习的参数 w，初始值为随机数
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        """
        前向传播函数
        - queries: 查询（新输入）
        - keys: 键（训练数据的输入）
        - values: 值（训练数据的输出）
        """
        # 将 queries 重复，使其形状与 keys 匹配
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        # 计算注意力权重，使用带参数的高斯核
        self.attention_weights = nn.functional.softmax(-1 / 2 * ((queries - keys) * self.w) ** 2, dim=1)
        # 对 values 进行加权平均，得到预测值
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)


# 生成示例数据
# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
print(X_tile, X_tile.shape)
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
print(keys)
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

# 初始化模型
net = NWKernelRegression()
# 定义损失函数（均方误差）
loss = nn.MSELoss(reduction='none')
# 定义优化器（随机梯度下降）
trainer = torch.optim.SGD(net.parameters(), lr=0.05)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])
# 训练模型
for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
animator.show()

# # keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
# keys = x_train.repeat((n_test, 1))
# # value的形状:(n_test，n_train)
# values = y_train.repeat((n_test, 1))
# y_hat = net(x_test, keys, values).unsqueeze(1).detach()
#
# # y_hat = torch.matmul(attention_weights, y_truth)
# plot_kernel_reg(y_hat)
# d2l.plt.show()
#
# d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
#                   xlabel='Sorted training inputs',
#                   ylabel='Sorted testing inputs')
# d2l.plt.show()
