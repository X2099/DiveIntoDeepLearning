# -*- coding: utf-8 -*-
"""
@File    : d2l.py
@Time    : 2024/3/5 11:49
@Desc    : 
"""
import os
import time
import torch
from torch.utils import data
import torchvision
from torchvision import transforms
from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def use_svg_display():
    """
    使用svg格式在Jupyter中显示绘图
    """
    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(7, 5)):
    """
    设置matplotlib的图表大小
    """
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """
    设置matplotlib的轴
    """
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(7, 5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


class Timer:  # @save
    """记录多次运行时间"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


def synthetic_data(w, b, num_examples):
    """
    生产数据：y=Xw+b+噪声
    :param w: 权重向量
    :param b: 偏置向量
    :param num_examples: 样本数量
    :return:
    """
    X = torch.normal(0, 1, (num_examples, len(w)))  # 按正态分布随机生成
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 噪声
    return X, y.reshape(-1, 1)


def linreg(X, w, b):
    """
    线性回归模型
    :param X: 样本矩阵
    :param w: 权重
    :param b: 编制
    :return: 预测值
    """
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """
    平方损失函数
    :param y_hat: 预测值
    :param y: 标签
    :return: 损失
    """
    return 1 / 2 * (y_hat - y) ** 2


def sgd(params, lr, batch_size):
    """
    小批量随机梯度下降
    :param params: 参数
    :param lr: 学习率
    :param batch_size: 批量大小
    :return:
    """
    with torch.no_grad():  # 在这个上下文环境中，不进行梯度计算
        for param in params:
            # 这里的梯度是对本批所有样本梯度的总和，所以要除以批量大小
            param -= lr * param.grad / batch_size
            param.grad.zero_()  # 将参数梯度清零


def load_array(data_arrays, batch_size, is_train=True):
    """
    小批量读取数据集
    :param data_arrays: 总样本
    :param batch_size: 批量大小
    :param is_train: 是否训练
    :return:
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    # text_labels = [
    #     't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    # ]
    text_labels = ['T恤', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '短靴']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


def load_data_fashion_mnist(batch_size, resize=None):
    """
    加载数据
    :param batch_size:
    :param resize:
    :return:
    """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=False)
    train_data = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
    test_data = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers())
    return train_data, test_data


if __name__ == '__main__':
    import numpy as np

    x = np.arange(0, 10, 0.1)
    # plot(x, x ** 2, legend=['x的平方'])
    # plt.show()
    plt.plot(x, label='123')
    plt.show()
