# -*- coding: utf-8 -*-
"""
@File    : 3.5.图像分类数据集.py
@Time    : 2024/11/11 14:36
@Desc    : 
"""
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import d2l

# 图像数据转换为浮点数并标准化
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

print(len(mnist_train), len(mnist_test))  # 60000 10000
print(mnist_train[0][0].shape)  # torch.Size([1, 28, 28])


def get_fashion_mnist_labels(labels):
    text_labels = ["T恤", "裤子", "套衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "短靴"]
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""

    # 设置显示图像的整体尺寸，行数和列数乘以比例来调整每个子图的大小
    figsize = (num_cols * scale, num_rows * scale)
    # 创建子图矩阵，返回一个包含多个轴对象的数组，用于绘制每个图像
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    # 将多维的轴对象数组展平，以方便后续循环中逐一访问每个子图的轴
    axes = axes.flatten()
    # 遍历图像和对应的轴对象，将每张图像显示在各自的子图中
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        # 如果图像是 torch 张量格式，将其转换为 numpy 格式，以适应 imshow 的展示要求
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)  # 否则直接展示图像（假设是 PIL 格式）
        # 隐藏子图的 X 轴和 Y 轴坐标
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        # 如果提供了标题列表，设置每张图像的标题
        if titles:
            ax.set_title(titles[i])
    d2l.plt.show()
    return axes  # 返回轴对象数组，便于进一步的操作或调整


X, y = next(iter(data.DataLoader(mnist_train, batch_size=25)))
show_images(X.reshape(25, 28, 28), 5, 5, titles=get_fashion_mnist_labels(y))

batch_size = 256


def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 0


train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())

timer = d2l.Timer()
for X, y in train_iter:
    pass
print(f'{timer.stop():.2f} sec')  # 2.49 sec

train_iter, test_iter = d2l.load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype)  # torch.Size([32, 1, 64, 64]) torch.float32
    print(y.shape, y.dtype)  # torch.Size([32]) torch.int64
    break
