# -*- coding: utf-8 -*-
"""
@File    : 2.2.数据预处理.py
@Time    : 2024/10/4 10:28
@Desc    : 
"""
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# 读取 CSV 数据
data = pd.read_csv('data.csv')

# 打印前5条数据
# print(data.head())

# 将 Pandas 数据框转换为 PyTorch 张量
features = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
labels = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32)


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 创建数据集
dataset = CustomDataset(features, labels)

# 使用 DataLoader 进行批量处理
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # 打印一个批次的数据
# for batch_features, batch_labels in dataloader:
#     print(batch_features, batch_labels)
#     break  # 只打印一个批次

# 定义简单的模型
model = torch.nn.Linear(3, 1)

print(f"model: {model}")
"""
model: Linear(in_features=3, out_features=1, bias=True)
"""

# 定义损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
loss = None
for epoch in range(5):
    for batch_features, batch_labels in dataloader:
        # 前向传播
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels.view(-1, 1))
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/5], Loss: {loss.item():.4f}')

"""
Epoch [1/5], Loss: 4.8706
Epoch [2/5], Loss: 0.0903
Epoch [3/5], Loss: 0.0662
Epoch [4/5], Loss: 0.0620
Epoch [5/5], Loss: 0.0581
"""