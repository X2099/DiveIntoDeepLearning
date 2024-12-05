# -*- coding: utf-8 -*-
"""
@File    : 3.7.Softmax回归的简洁实现.py
@Time    : 2024/11/13 15:32
@Desc    : 
"""
import torch.optim
from torch import nn
import d2l

# 设置批量大小
batch_size = 256
# 加载训练和测试数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

in_features = 28 * 28  # 输入特征数（展平成一维）
out_features = 10  # 输出特征数
# 使用框架定义模型
net = nn.Sequential(nn.Flatten(), nn.Linear(in_features, out_features))


# 初始化模型权重参数
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)


# 模型应用参数
net.apply(init_weights)

# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')

lr = 0.1  # 学习率
# 使用小批量随机梯度下降（SGD）作为优化算法
trainer = torch.optim.SGD(net.parameters(), lr=lr)

if __name__ == '__main__':
    num_epochs = 10
    metric = d2l.Accumulator(3)
    animator = d2l.Animator(xlabel='轮数', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'], title="softmax回归模型训练效果图")
    # 训练模型
    d2l.train_softmax(net, train_iter, test_iter, loss,
                      num_epochs, trainer, batch_size, animator)
    # 可视化优化过程
    animator.show()
