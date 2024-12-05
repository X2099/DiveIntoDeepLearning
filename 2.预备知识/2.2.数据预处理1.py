# -*- coding: utf-8 -*-
"""
@File    : 2.2.数据预处理1.py
@Time    : 2024/10/4 20:19
@Desc    : 
"""
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 定义数据增强操作
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转10度
    transforms.ToTensor()  # 转换为张量
])

# 加载并增强图像数据
from torchvision.datasets import CIFAR10

"""
CIFAR-10 是计算机视觉领域中常用的数据集之一，主要用于图像分类任务。
它包含 10 类不同类型的彩色图像，数据集的总规模为 60,000 张 32x32 像素的彩色图像，
其中训练集包含 50,000 张图像，测试集包含 10,000 张图像。
"""
train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 打印增强后的图像数据
data_iter = iter(train_loader)
images, labels = next(data_iter)

print(images.shape)
"""
torch.Size([32, 3, 32, 32])
"""
