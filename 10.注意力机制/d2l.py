# -*- coding: utf-8 -*-
"""
@File    : d2l.py
@Time    : 2024/10/24 10:02
@Desc    : 
"""
import hashlib
import math
import random
import re
import sys
import os
import tarfile
import time
import zipfile
import collections
from itertools import islice
from pprint import pprint

import jieba
import numpy as np
import requests
import torchvision.datasets
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from matplotlib import rcParams

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 设置中文字体（SimHei 是黑体字）
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
# rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
# plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示成方块的问题

d2l = sys.modules[__name__]


def use_svg_display():  # @save
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')


# 我们定义set_figsize函数来设置图表大小。
# 注意，这里可以直接使用d2l.plt，因为导入语句 from matplotlib import pyplot as plt已标记为保存到d2l包中。
def set_figsize(figsize=(3.5, 2.5)):  # @save
    """设置matplotlib的图表大小"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize


# 下面的set_axes函数用于设置由matplotlib生成图表的轴的属性。
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


# 通过这三个用于图形配置的函数，定义一个plot函数来简洁地绘制多条曲线，因为我们需要在整个书中可视化许多曲线。
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(6.5, 5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

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
    plt.show()


def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, [num_examples, len(w)], requires_grad=True)
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - torch.reshape(y, y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


class Timer:
    """记录多次运行时间"""

    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
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


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def accuracy(y_hat, y):
    # if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
    #     y_hat = y_hat.argmax(dim=1)
    # cmp = y_hat.type(y_hat.dtype) == y
    # return float(cmp.type(y.dtype).sum())
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y_hat.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        # 将输入图像的高度和宽度调整为指定的尺寸 resize，从而确保图像的大小一致
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)  # 将多个图像转换操作组合在一起，以形成一个“转换流水线”
    mnist_train = torchvision.datasets.FashionMNIST(root="../data",
                                                    train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data",
                                                   train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False))


class Animator:
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5), title=None):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.title = title  # 新增标题属性
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        if self.title:  # 如果存在标题，就设置标题
            self.axes[0].set_title(self.title)
        self.config_axes()

    def show(self):
        # 显示图形
        # print(plt.rcParams['font.family'])  # 查看当前字体
        plt.show()
        plt.savefig('output.png')


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
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
    d2l.plt.show()
    return axes


def get_fashion_mnist_labels(labels):
    text_labels = ["T恤", "裤子", "套衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "短靴"]
    return [text_labels[int(i)] for i in labels]


def train_epoch(net, train_iter, loss, updater, batch_size):
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(batch_size)
        metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, batch_size):
    animator = Animator(xlabel='轮数', xlim=[1, num_epochs], ylim=[0, 1],
                        legend=['train loss', 'train acc', 'test acc'], title="PyTorch高级API实现Dropout的MLP")
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater, batch_size)
        test_acc = d2l.evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        print(f'epoch {epoch + 1}, train loss {train_metrics[0]:.3f}, '
              f'train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}')
    # 可视化优化过程
    animator.show()


def predict_ch3(net, test_iter, n=6):
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = ["真实：" + true + '\n' + "预测：" + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n], scale=2.5)


def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        print(l.mean())
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset.

    Defined in :numref:`sec_utils`"""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def download(name, cache_dir=os.path.join('..', 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1_hash == sha1.hexdigest():
            return fname
    print(f"正在从{url}下载{fname}")
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)


def try_gpu(i=0):
    """
    如果存在，则返回gpu(i)，否则返回cpu()
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """
    返回所有可用的GPU，如果没有GPU，则返回[cpu(),]
    """
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())
               ]
    return devices if devices else [torch.device('cpu'), ]


def corr2d(X, K):
    """
    互相关运算实现
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """
    使用GPU计算模型在数据集上的精度
    """
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            """
            这行代码的目的是确定模型的第一个参数（通常是权重）被存储在哪个设备上。
            通常这个信息是用于确定当前模型运行在 CPU 还是 GPU 上，
            在使用 GPU 进行训练时非常有用，特别是当模型和数据需要显式迁移到同一个设备上时。
            """
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            # 采用 均匀分布 来初始化权重
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel="epoch", xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'],
                            figsize=(9, 5), title="Designing Convolution Network (DenseNet)")
    timer, num_batches = d2l.Timer(), len(train_iter)
    metric = d2l.Accumulator(3)
    train_l, train_acc, test_acc = 0, 0, 0
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
              f'on {str(device)}')
        print('-' * 20 + f' 第{epoch + 1}轮训练 ' + '-' * 20)
    animator.show()


DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce'
)

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90'
)


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(5, 5),
                  cmap='Reds'):
    """显示矩阵热图的函数

    参数：
    matrices: 形状为 (num_rows, num_cols, height, width) 的四维张量，每个子矩阵表示一个热图
    xlabel: 横轴标签
    ylabel: 纵轴标签
    titles: 每个子图的标题，默认为 None
    figsize: 整个热图的尺寸，默认为 (2.5, 2.5)
    cmap: 颜色映射方案，默认为 'Reds'
    """

    # 使用 SVG 显示，以获得更清晰的图片
    d2l.use_svg_display()
    # 获取矩阵的行数和列数（即热图的排列方式）
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    # 创建子图网格，sharex 和 sharey 使所有子图共享坐标轴，squeeze=False 保证 axes 仍然是 2D 结构
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)

    # 遍历所有行和列，绘制每个小热图
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            # 将矩阵数据转换为 NumPy 数组并显示为热图
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            # 设置横轴标签，只在最后一行的子图显示
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            # 设置纵轴标签，只在第一列的子图显示
            if j == 0:
                ax.set_ylabel(ylabel)
            # 设置子图的标题（如果提供了标题列表）
            if titles:
                ax.set_title(titles[j])

    # 添加颜色条（colorbar）以指示热图的数值强度
    fig.colorbar(pcm, ax=axes, shrink=0.6)


DATA_HUB['time_machine'] = (
    DATA_URL + 'timemachine.txt',
    '090b5e7e70c295757f55df93cb0a180b9691891a'
)


def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        raise Exception(f'错误：未知词元类型：{token}')


class Vocab:
    """文本词表，用于将词元（token）映射为数字索引"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        初始化词表，构建词元到数字索引的映射

        参数:
        - tokens (list): 词元的列表，可以是文本数据中提取出的词元。每个词元可以是单词或字符，默认值为 `None`。
        - min_freq (int): 最小词频，只有词频大于或等于该值的词元才会被加入词表，默认值为 `0`。
        - reserved_tokens (list): 预留词元的列表，这些词元会优先添加到词表中（例如：`<unk>`、`<pad>`等）。默认值为 `None`。
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 统计词元频率，并按频率降序排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 初始化词表，<unk>代表未知词元
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        # 根据频率添加词元，忽略低频词
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        """返回词表大小"""
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """
        根据词元返回对应的索引，支持列表输入

        参数:
        - tokens (str, list, tuple): 输入一个词元（字符串）或者词元列表（字符串列表）。
          如果是单个词元，返回该词元的索引；如果是列表或元组，返回每个词元的索引列表。

        返回:
        - (int, list): 如果输入是单个词元，返回对应的索引；如果是词元列表，返回一个索引列表。
        """
        if not isinstance(tokens, (tuple, list)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        """返回未知词元的索引（0）"""
        return 0

    @property
    def token_freqs(self):
        """返回词元频率列表"""
        return self._token_freqs


def count_corpus(tokens):
    """
    统计词元频率，支持1D和2D列表

    参数:
    - tokens (list, list of lists): 输入词元列表，可以是一个一维的词元列表，或者是多个文本行的词元列表（二维列表）。

    返回:
    - (dict): 返回一个字典，键为词元，值为该词元在语料库中出现的频率。
    """
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def load_corpus_time_machine(max_tokens=-1):
    """
    返回时光机器数据集的词元索引列表和词表

    参数:
    - max_tokens (int): 限制返回的词元数量。如果为负值，表示不限制数量，返回全部数据。默认值为 -1。

    返回:
    - corpus (list): 词元索引的列表，表示文本中的每个字符（词元）的索引。
    - vocab (Vocab): 词表对象，用于映射词元和索引之间的关系。
    """
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')  # 使用字符级别的词元化
    vocab = Vocab(tokens)  # 构建词表
    # 将所有文本行展平为一个词元索引的列表
    corpus = [vocab[token] for line in tokens for token in line]
    # 如果限制了词元数量，则截取前 max_tokens 个词元
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):
    """
    使用随机抽样生成一个小批量子序列。

    参数:
    corpus (list): 输入的文本数据（词元列表）。
    batch_size (int): 每个批次的样本数量。
    num_steps (int): 每个序列的长度。

    返回:
    iterator: 返回一个生成器，每次迭代返回一个批次的输入序列（X）和目标序列（Y）。
    """
    # 从随机偏移量开始对序列进行分区，随机范围包括 num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 计算可以从文本中划分出的子序列数量，减去1是因为要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 获取每个子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 随机打乱子序列的起始索引
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从 pos 位置开始的长度为 num_steps 的子序列
        return corpus[pos:pos + num_steps]

    # 计算每个批次的子序列数量
    num_batches = num_subseqs // batch_size
    # 迭代批次
    for i in range(0, batch_size * num_batches, batch_size):
        # 获取当前批次的随机起始索引
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        # 生成当前批次的输入序列 X 和目标序列 Y
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]

        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    # 确保每个批次的大小都是batch_size的整数倍
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    # 创建输入（Xs）和标签（Ys）序列
    Xs = torch.tensor(corpus[offset:offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])
    # 重塑成 (batch_size, num_steps) 的形状
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    # 计算可以生成的批次数量
    num_batches = Xs.shape[1] // num_steps
    # 生成每个小批量的数据
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y


class SeqDataLoader:
    """加载序列数据的迭代器"""

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        # 根据选择的迭代方式设置对应的数据迭代函数
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        # 加载数据
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        # 保存批大小和时间步数
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


class RNNModelScratch:
    """从零开始实现的循环神经网络模型，用于字符级文本生成。"""

    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        """
        初始化 RNN 模型。

        参数:
        vocab_size (int): 词汇表大小。
        num_hiddens (int): 隐藏层单元数。
        device (torch.device): 设备。
        get_params (function): 获取模型参数的函数。
        init_state (function): 初始化隐藏状态的函数。
        forward_fn (function): 前向传播函数。
        """
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        # 获取模型的所有参数
        self.params = get_params(vocab_size, num_hiddens, device)
        # 设置初始化状态函数和前向传播函数
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        """
        调用模型进行前向传播。

        参数:
        X (Tensor): 输入数据，形状为 (batch_size, seq_len)。
        state (Tensor): 上一时刻的隐藏状态，形状为 (batch_size, num_hiddens)。

        返回:
        输出结果和更新后的隐藏状态。
        """
        # 将输入转换为one-hot编码，并进行类型转换
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)  # 调用前向传播函数进行计算

    def begin_state(self, batch_size, device):
        """
        初始化隐藏状态。

        参数:
        batch_size (int): 输入数据的批量大小。
        device (torch.device): 设备类型（'cpu' 或 'gpu'）。

        返回:
        Tensor: 初始化的隐藏状态，形状为 (batch_size, num_hiddens)。
        """
        return self.init_state(batch_size, self.num_hiddens, device)


def predict_ch8(prefix, num_preds, net, vocab, device):
    """
    使用训练好的RNN模型生成文本。

    参数:
    prefix (str): 输入的文本前缀，模型将基于这个前缀生成后续文本。
    num_preds (int): 要生成的字符数。
    net (nn.Module): 训练好的RNN模型。
    vocab (Vocab): 词汇表，用于处理输入和输出的字符。
    device (torch.device): 设备（CPU或GPU），决定模型和数据存放的位置。

    返回:
    str: 生成的文本序列。
    """
    state = net.begin_state(batch_size=1, device=device)  # 初始状态为空
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # 将前缀转换为模型的输入，获取对应的字符索引
    for y in prefix[1:]:
        # 使用模型预测下一个字符，更新隐状态
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    # 生成num_preds个字符
    for _ in range(num_preds):
        # 获取当前模型的输出（预测的下一个字符）
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))

    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    """裁剪梯度
    对模型的梯度进行裁剪，防止梯度爆炸。
    参数：
        net: 神经网络模型，可以是 `nn.Module` 的实例或包含参数的自定义对象。
        theta: 裁剪阈值。如果梯度的范数超过该阈值，进行裁剪。
    """
    # 如果 net 是 nn.Module 类的实例，获取其所有可训练的参数
    if isinstance(net, nn.Module):
        # 获取所有需要梯度更新的参数
        params = [param for param in net.parameters() if param.requires_grad]
    else:
        # 如果 net 不是 nn.Module 实例，假设它有一个 `params` 属性
        params = net.params
    # 计算所有参数的梯度范数（L2范数）：对所有参数的梯度进行平方和，再取平方根
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    # 如果梯度的范数超过了阈值 theta，就进行裁剪
    if norm > theta:
        for param in params:
            # 通过缩放梯度，确保范数不超过 theta
            param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期
    参数：
        net: 待训练的网络模型
        train_iter: 训练数据迭代器
        loss: 损失函数
        updater: 更新器，可以是优化器或自定义的更新方法
        device: 计算设备（如 'cpu' 或 'cuda'）
        use_random_iter: 是否使用随机抽样
    返回：
        返回训练损失的指数损失（即对数损失的指数）和每秒处理的样本数量
    """
    state, timer = None, Timer()  # 初始化模型状态和计时器
    metric = Accumulator(2)  # 用于累积训练损失和词元数量的辅助器
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 如果是第一次迭代或者需要随机抽样时，初始化模型的隐状态
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # 如果模型是nn.Module且隐状态是一个张量（如GRU）
                state.detach_()  # 将隐状态从计算图中分离，避免梯度传播
            else:
                # 如果模型是LSTM或自定义模型，隐状态可能是一个元组
                for s in state:
                    s.detach_()  # 将每个隐状态分离
        y = Y.T.reshape(-1)  # 转置并拉直标签张量
        X, y = X.to(device), y.to(device)  # 将输入和标签移动到指定设备
        # 前向传播：计算预测结果和更新隐状态
        y_hat, state = net(X, state)
        # 计算当前批次的损失，使用long类型的标签
        l = loss(y_hat, y.long()).mean()

        if isinstance(updater, torch.optim.Optimizer):
            # 如果updater是优化器对象（如torch.optim.Optimizer）
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)  # 梯度裁剪，避免梯度爆炸
            updater.step()
        else:
            # 如果updater是自定义的更新方法
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        # 累积当前批次的损失和词元数量
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型
    参数：
        net: 待训练的网络模型
        train_iter: 训练数据迭代器
        vocab: 词汇表
        lr: 学习率
        num_epochs: 训练的周期数
        device: 计算设备（如 'cpu' 或 'cuda'）
        use_random_iter: 是否使用随机抽样（默认为 False）
    """
    loss = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    animator = Animator(xlabel='epoch', ylabel='perplexity',
                        legend=['train'], xlim=[10, num_epochs],
                        figsize=(6.18, 3.82))
    # 初始化更新器
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)

    # 定义预测函数，用于根据给定前缀生成文本
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        # 在每个周期中训练并计算困惑度和每秒处理的词元数
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        # 每经过10个周期打印一次预测结果
        if (epoch + 1) % 10 == 0:
            print(epoch + 1, predict('time traveller'))  # 以'time traveller'为前缀进行预测
            animator.add(epoch + 1, [ppl])

    # 打印最终的困惑度和处理速度
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')

    # 打印以'time traveller'和'traveller'为前缀的预测文本
    print(predict('time traveller'))
    print(predict('traveller'))
    animator.show()


class RNNModel(nn.Module):
    """循环神经网络模型"""

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后会介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(dtype=torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为 (时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU（门控循环单元，RNN的一种变体）以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (
                torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                            device=device),
                torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                            device=device)
            )


DATA_HUB['fra-eng'] = (DATA_URL + 'fra-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5')


def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()


def preprocess_nmt(text: str):
    """预处理“英语－法语”数据集"""

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    set_figsize((7, 4.82))
    _, _, patches = plt.hist([[len(l) for l in xlist], [len(l) for l in ylist]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    plt.legend(legend)
    plt.show()


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[line] for line in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(dim=1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        # X 为输入序列（长度可变）
        # 此处应实现编码逻辑，将 X 转换为固定形状的编码状态
        raise NotImplementedError


class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""

    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        # enc_X：输入序列
        # dec_X：解码器接收的输入（例如目标序列的部分）
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


class Seq2SeqEncoder(Encoder):
    """用于序列到序列学习的循环神经网络编码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


class Seq2SeqDecoder(Decoder):
    """用于序列到序列学习的循环神经网络解码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), dim=2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange(maxlen, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""

    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)

    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        elif type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if 'weight' in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = Animator(xlabel='epoch', ylabel='loss',
                        xlim=[10, num_epochs], figsize=(6.18, 3.82))
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # 损失函数的标量进行“反向传播”
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
            print(f'epoch {epoch + 1} loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
                  f'tokens/sec on {str(device)}')

    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device),
        dim=0
    )
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(
        torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device),
        dim=0
    )
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k):
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def masked_softmax(X, valid_lens):
    """
    执行掩蔽 Softmax 操作。

    该函数会根据有效长度 `valid_lens` 对输入张量 `X` 进行掩蔽处理。掩蔽部分将被置为一个非常小的负数（例如 -1e6），
    以确保它们在 Softmax 操作中得到零的权重。

    参数：
    X (Tensor): 需要执行 Softmax 的输入张量。通常是一个 3D 张量，形状为 (batch_size, seq_len, num_classes)。
    valid_lens (Tensor): 有效长度张量。它指定了每个序列中有效的元素的个数。它的形状可以是一个 1D 张量，表示
                          每个序列的有效长度，或者是一个 2D 张量，每行表示对应样本中每个查询的有效长度。

    返回：
    Tensor: 执行掩蔽 Softmax 后的结果。与输入 `X` 形状相同，但掩蔽部分的权重会变成零。
    """
    if valid_lens is None:
        return F.softmax(X, dim=-1)

    # 获取 X 的形状 (batch_size, seq_len, num_classes)
    shape = X.shape
    if valid_lens.dim() == 1:
        # 如果 valid_lens 是一维的，将它扩展为二维，重复有效长度，适配每个序列
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        # 确保 valid_lens 是一个一维张量
        valid_lens = valid_lens.reshape(-1)
    # 将 X 展平，变成二维张量 (batch_size * seq_len, num_classes)
    X = X.reshape(-1, shape[-1])
    # 创建掩蔽操作，超出有效长度的位置被替换为一个非常小的负数（-1e6）
    X = torch.where(
        valid_lens.unsqueeze(-1) > torch.arange(X.shape[-1], device=X.device),
        X,  # 对应有效位置保留原值
        torch.tensor(-1e6)  # 对于无效位置，使用极小值 -1e6
    )
    # 执行 Softmax 操作，返回 Softmax 的结果，dim=-1 表示按最后一个维度（num_classes）计算 Softmax
    return F.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    """加性注意力（Additive Attention）类"""

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        # 定义键（key）的线性变换层，将输入的键映射到隐藏层空间
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        # 定义查询（query）的线性变换层，将输入的查询映射到隐藏层空间
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        # 定义一个线性变换层，用于将注意力得分计算为一个标量
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        """
        前向传播方法，计算加性注意力的输出。

        参数：
        查询、键和值的形状为（批量大小，步数或词元序列长度，特征大小）
        queries: 查询张量，形状为 (batch_size, query_len, query_dim)
        keys: 键张量，形状为 (batch_size, key_len, key_dim)
        values: 值张量，形状为 (batch_size, key_len, value_dim)
        valid_lens: 有效长度，表示哪些键是有效的，通常用于掩蔽填充部分，形状为 (batch_size, query_len)

        返回值：
        注意力汇聚输出的形状为（批量大小，查询的步数，值的维度）
        返回加权后的值张量，形状为 (batch_size, query_len, value_dim)
        """
        # 将查询和键分别映射到隐藏空间
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 进行广播求和：查询和键的每一对进行加和，形状变为 (batch_size, query_len, key_len, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        # 通过tanh激活函数增加非线性变换
        features = torch.tanh(features)
        # 使用w_v计算加性注意力得分。输出形状为 (batch_size, query_len, key_len)
        scores = self.w_v(features).squeeze(-1)
        # 计算注意力权重，使用softmax进行归一化，注意masking掉无效位置
        self.attention_weights = masked_softmax(scores, valid_lens)
        # 使用注意力权重对值进行加权求和，得到最终的加性注意力输出
        # 使用 dropout 防止过拟合
        return torch.bmm(self.dropout(self.attention_weights), values)


class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.attention = AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens,
                          num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return outputs.permute(1, 0, 2), hidden_state, enc_valid_lens

    def forward(self, X, state):
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size,
        # num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出X的形状为(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # query的形状为(batch_size,1,num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context的形状为(batch_size,1,num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特征维度上连结
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]  # 查询和键的维度

        # 计算查询和键的点积，注意进行缩放
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)

        # 计算注意力权重
        self.attention_weights = masked_softmax(scores, valid_lens)

        # 返回加权求和的值
        return torch.bmm(self.dropout(self.attention_weights), values)


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""

    # 输入X的形状: (batch_size, 查询或者“键－值”对的个数, num_hiddens)
    # 输出X的形状: (batch_size, 查询或者“键－值”对的个数, num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 输出X的形状: (batch_size, num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 最终输出的形状: (batch_size*num_heads,查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""

    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)

        # valid_lens　的形状:
        # (batch_size,)或(batch_size, 查询的个数)

        # 经过变换后，输出的queries, keys, values　的形状:
        # (batch_size*num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)

        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状: (batch_size*num_heads, 查询的个数, num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size, 查询的个数, num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络（Position-wise Feed-Forward Network, FFN）

    该网络用于 Transformer 结构中的每个位置的前馈计算，不同于传统的全连接网络，
    它对序列的每个时间步独立处理，不共享参数。通常由两个线性变换和一个 ReLU 激活函数组成。
    """

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        """
        初始化前馈网络层。

        参数：
        ffn_num_input (int): 输入特征的维度。
        ffn_num_hiddens (int): 隐藏层的特征维度。
        ffn_num_outputs (int): 输出特征的维度。
        """
        super().__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        """
        前向传播计算。

        参数：
        X (Tensor): 输入张量，形状通常为 (batch_size, seq_len, ffn_num_input)。

        返回：
        Tensor: 输出张量，形状为 (batch_size, seq_len, ffn_num_outputs)。
        """
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接后进行层规范化（Add & Norm）"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        """
        初始化 AddNorm 模块。

        参数：
        normalized_shape (int or tuple): 需要进行层规范化的特征维度。
        dropout (float): Dropout 的丢弃率，用于防止过拟合。
        """
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)  # 施加 Dropout，防止过拟合
        self.ln = nn.LayerNorm(normalized_shape)  # 进行层规范化

    def forward(self, X, Y):
        """
        前向传播：执行残差连接（X + dropout(Y)）并进行层规范化。

        参数：
        X (Tensor): 残差连接的原始输入。
        Y (Tensor): 需要施加 Dropout 并参与残差连接的输出。

        返回：
        Tensor: 经过残差连接和层规范化后的结果。
        """
        return self.ln(X + self.dropout(Y))


class EncoderBlock(nn.Module):
    """Transformer 编码器块

    该模块包括多头自注意力机制、残差连接、层归一化以及前馈神经网络。
    """

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        """
        参数：
        key_size: int，键的维度
        query_size: int，查询的维度
        value_size: int，值的维度
        num_hiddens: int，隐藏层维度
        norm_shape: tuple，层归一化的形状
        ffn_num_input: int，前馈神经网络的输入维度
        ffn_num_hiddens: int，前馈神经网络的隐藏层维度
        num_heads: int，多头注意力的头数
        dropout: float，Dropout 概率
        use_bias: bool，是否在注意力计算时使用偏置
        """
        super().__init__(**kwargs)
        # 多头自注意力机制
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                            num_heads, dropout, use_bias)
        # 第一层 AddNorm（用于多头注意力后的归一化）
        self.addnorm1 = AddNorm(norm_shape, dropout)
        # 前馈神经网络（FFN）
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        # 第二层 AddNorm（用于 FFN 之后的归一化）
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        """
        前向传播过程：
        1. 进行多头自注意力计算，并通过 AddNorm 归一化
        2. 经过前馈神经网络，并通过 AddNorm 归一化

        参数：
        X: 张量，输入序列，形状为 (batch_size, num_tokens, num_hiddens)
        valid_lens: 张量，掩码长度，防止模型关注填充部分

        返回：
        归一化后的输出张量，形状与 X 相同
        """
        # 多头自注意力 + 残差连接 + 归一化
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        # 前馈神经网络 + 残差连接 + 归一化
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(Encoder):
    """Transformer 编码器

    该编码器由嵌入层（Embedding）、位置编码（Positional Encoding）、
    多个 Transformer 编码器块（EncoderBlock）组成。
    """

    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        """
        参数：
        vocab_size: int，词汇表大小
        key_size: int，键（key）的维度
        query_size: int，查询（query）的维度
        value_size: int，值（value）的维度
        num_hiddens: int，隐藏单元的维度
        norm_shape: tuple，层规范化的形状
        ffn_num_input: int，前馈神经网络的输入维度
        ffn_num_hiddens: int，前馈神经网络的隐藏层维度
        num_heads: int，多头注意力的头数
        num_layers: int，编码器块的层数
        dropout: float，Dropout 概率
        use_bias: bool，是否在注意力计算中使用偏置
        """
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens

        # 词嵌入层：将输入索引映射到指定维度的嵌入向量
        self.embedding = nn.Embedding(vocab_size, num_hiddens)

        # 位置编码：引入位置信息，使模型能够识别序列顺序
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)

        # 由多个 Transformer 编码器块组成的序列
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                'block' + str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias)
            )

    def forward(self, X, valid_lens, *args):
        """
        前向传播：
        1. 计算嵌入表示，并进行缩放
        2. 加入位置编码，使模型具有顺序信息
        3. 通过多个 Transformer 编码器块处理输入
        4. 记录每层的注意力权重

        参数：
        X: 张量，输入序列，形状为 (batch_size, num_tokens)
        valid_lens: 张量，有效序列长度，用于掩蔽填充部分

        返回：
        处理后的张量，形状为 (batch_size, num_tokens, num_hiddens)
        """
        # 计算词嵌入，并进行缩放，使其数值范围更适合位置编码
        # 由于位置编码的值在 [-1, 1] 之间，因此嵌入乘以 sqrt(num_hiddens)
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X


class DecoderBlock(nn.Module):
    """解码器中第 i 个块：包括两次多头注意力和一层前馈神经网络"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        """
        初始化解码器块

        参数：
        key_size, query_size, value_size: 注意力机制中的键、查询和值的维度
        num_hiddens: 多头注意力和前馈网络的输出维度
        norm_shape: 层归一化的输入形状
        ffn_num_input: 前馈神经网络的输入维度
        ffn_num_hiddens: 前馈神经网络的隐藏层维度
        num_heads: 注意力头的数量
        dropout: Dropout 概率
        i: 解码器块的索引（用于状态的存取）
        """
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        # 第一个多头注意力：对解码器输入做掩蔽自注意力（masked self-attention）
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        # 第二个多头注意力：对编码器输出做注意力（encoder-decoder attention）
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        # 前馈网络部分
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        """
        前向传播

        参数：
        X: 当前解码器的输入，形状 (batch_size, tgt_seq_len, num_hiddens)
        state:
            state[0]: 编码器输出，用作第二个注意力层的键和值
            state[1]: 编码器的有效长度，用于掩蔽 padding
            state[2]: 解码器中每个块的缓存 key/value，用于预测时自回归推理

        返回：
        输出 (batch_size, tgt_seq_len, num_hiddens)，以及更新后的 state
        """
        enc_outputs, enc_valid_lens = state[0], state[1]
        # key_values 缓存自注意力历史信息（用于预测阶段）
        if state[2][self.i] is None:
            key_values = X  # 初始化：当前时间步的输入即为 key 和 value
        else:
            # 累积历史信息：拼接当前时间步输入和已有 key_values
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values  # 更新缓存

        if self.training:
            # 训练阶段：构造解码器注意力掩码（避免未来信息泄露）
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            # 推理阶段：逐步解码，因此无需掩蔽
            dec_valid_lens = None

        # 第一个多头注意力：对解码器当前输入进行掩蔽自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)  # 残差连接 + 层归一化

        # 第二个多头注意力：解码器对编码器输出进行注意力
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)  # 残差连接 + 层归一化

        # 前馈网络处理 + 残差连接 + 层归一化
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(AttentionDecoder):
    """Transformer 解码器"""

    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        """
        初始化 Transformer 解码器

        参数说明：
        vocab_size: 词表大小
        key_size, query_size, value_size: 多头注意力中键、查询、值的维度
        num_hiddens: 隐藏单元数，也是词嵌入和位置编码的维度
        norm_shape: 层归一化的输入形状
        ffn_num_input: 前馈神经网络输入维度
        ffn_num_hiddens: 前馈神经网络隐藏层维度
        num_heads: 注意力头数
        num_layers: 解码器块的数量
        dropout: dropout 比例
        """
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers

        # 词嵌入层，将输入 token 转换为向量
        self.embedding = nn.Embedding(vocab_size, num_hiddens)

        # 位置编码（带 dropout），加入序列位置信息
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)

        # 多个解码器块堆叠（每个块是 DecoderBlock）
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, i))

        # 输出层：将隐藏状态映射到词表大小
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        """
        初始化解码器状态

        返回值：
        - enc_outputs: 编码器输出
        - enc_valid_lens: 编码器输入的有效长度，用于掩蔽注意力
        - [None] * num_layers: 每个解码器块的自注意力缓存（预测时用）
        """
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        """
        前向传播

        参数：
        - X: 解码器输入（token ID 序列）
        - state: 编码器输出和解码器状态信息

        返回：
        - 输出 logits: 用于生成词的预测分布
        - 更新后的状态
        """
        # 嵌入并添加位置信息，然后缩放
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))

        # 初始化注意力权重存储列表（2层，每层包含每个 block 的权重）
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]

        # 依次通过每一个解码器块
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 记录自注意力权重（用于分析或可视化）
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # 记录编码器-解码器注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights

        # 通过输出层投影为词表大小，得到每个位置的预测结果
        return self.dense(X), state

    @property
    def attention_weights(self):
        """返回保存的注意力权重"""
        return self._attention_weights


def load_data_zh_en(batch_size, num_steps, n=10000):
    """返回中英翻译数据集的迭代器和词表"""
    data_zh_path = 'sample_data/news-commentary-v13.zh-en.zh'
    data_en_path = 'sample_data/news-commentary-v13.zh-en.en'
    with open(data_zh_path, 'r') as f:
        lines_zh = list(islice(f, n))
    with open(data_en_path, 'r') as f:
        lines_en = list(islice(f, n))
    tokens_zh = [jieba.lcut(line.strip().replace(' ', '')) for line in lines_zh]
    tokens_en = [re.findall(r"\w+|[^\w\s]", line) for line in lines_en]
    vocab_zh = d2l.Vocab(tokens_zh, min_freq=1, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    vocab_en = d2l.Vocab(tokens_en, min_freq=1, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(tokens_zh, vocab_zh, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(tokens_en, vocab_en, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, vocab_zh, vocab_en
