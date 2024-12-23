# -*- coding: utf-8 -*-
"""
@File    : 5.5. 读写文件.py
@Time    : 2024/12/23 10:36
@Desc    : 
"""
import torch
from torch import nn
from torch.nn import functional as F


# 创建一个张量
# X = torch.arange(4)

# print(X)  # tensor([0, 1, 2, 3])
# # 保存张量到文件
# torch.save(X, 'x-file')

# 加载存储的张量
# X2 = torch.load('x-file')
# print(X2)

# y = torch.zeros(4)
# 保存多个张量到文件
# torch.save([X, y], 'x-files')
# print(X, y)
# print('-' * 50)
# x2, y2 = torch.load('x-files')
# print(x2, y2)


# 创建一个包含张量的字典
# mydict = {'x': X, 'y': y}
# # 保存字典
# torch.save(mydict, 'mydict')
# # 加载字典
# mydict2 = torch.load('mydict')
# print(mydict2)


# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, X):
        return self.output(F.relu(self.hidden(X)))


# # 创建模型实例
net = MLP()
# 创建一些随机输入
X = torch.randn(size=(2, 20))
# 前向传播计算
Y = net(X)
print(Y.shape)
# """
# tensor([[ 0.1941, -0.0554,  0.2344,  0.4940,  0.1903,  0.0163,  0.1400,  0.2266,
#           0.1028, -0.3156],
#         [ 0.0139,  0.1429,  0.2452,  0.2173, -0.0184,  0.1851,  0.2068, -0.0041,
#          -0.1291, -0.1377]], grad_fn=<AddmmBackward0>)
# """
#
torch.save(net.state_dict(), 'mlp.params')

# 创建一个新的模型实例
clone = MLP()
# 加载保存的模型参数
clone.load_state_dict(torch.load('mlp.params', weights_only=True))
# 切换模型为评估模式
clone.eval()

# 验证两个模型输出是否一致
Y_clone = clone(X)
print(Y_clone.shape)
print(Y_clone == Y)
