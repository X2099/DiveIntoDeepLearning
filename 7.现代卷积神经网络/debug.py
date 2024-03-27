# -*- coding: utf-8 -*-
"""
@File    : 7.1.AlexNet.py
@Time    : 2024/3/22 17:08
@Desc    : 
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torchvision
from torchvision import transforms, models

from common.d2l import Timer

# 定义数据转换
data_transform = transforms.Compose([
    # 将图像调整为指定大小，以便与模型的输入尺寸匹配。这有助于确保模型能够接受统一大小的输入
    transforms.Resize(256),
    # 对图像进行中心裁剪，以去除图像边缘的无关信息。这在保留主要目标的同时减少了图像的大小
    transforms.CenterCrop(224),
    # 将图像转换为PyTorch张量格式，并对像素值进行归一化。这是因为PyTorch模型通常接受张量作为输入
    transforms.ToTensor(),
    # 对图像进行归一化处理，使得图像的像素值服从特定的分布，这有助于加速模型的收敛并提高训练效果
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 下载并加载CIFAR-10数据集
train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=data_transform)
test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=data_transform)

# 创建数据加载器
train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# 加载预训练的AlexNet模型
alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

# 将最后一层的全连接层替换成适合CIFAR-10数据集的新的全连接层
num_classes = 10  # 10个输出类别
alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)  # momentum参数引入了一个动量项

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
alexnet.to(device)

timer = Timer()  # 计时器

# 训练模型
num_epochs = 1
for epoch in range(num_epochs):
    running_loss = 0.0  # 损失
    correct = 0  # 预测正确的数量
    total = 0  # 总数量

    alexnet.train()

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = alexnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if i >= 200:
            print("用时：", timer.stop())
            break
    train_loss = running_loss / total
    train_acc = correct / total

    print(f'迭代周期 [{epoch + 1}/{num_epochs}], 训练损失: {train_loss:.4f}, 训练精度: {train_acc:.4f}')

import matplotlib.pyplot as plt
import numpy as np

# 在测试集上评估模型
alexnet.eval()
test_correct = 0
test_total = 0

# 在测试集上进行预测
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = alexnet(images)
        _, predicted = torch.max(outputs, 1)

        # 将张量转换为numpy数组
        images = images.cpu().numpy()
        labels = labels.cpu().numpy()
        predicted = predicted.cpu().numpy()

        # 可视化预测结果和图像
        for i in range(len(images)):
            plt.imshow(np.transpose(images[i], (1, 2, 0)))  # 将图像从(C, H, W)转换为(H, W, C)格式
            plt.title(f"Label: {labels[i]}, Predicted: {predicted[i]}")
            plt.show()
