# -*- coding: utf-8 -*-
"""
@File    : d2l.py
@Time    : 2024/3/5 11:49
@Desc    : 
"""
import os
import time
from IPython import display
import torch
from torch.utils import data
from torch import nn
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


def accuracy(y_hat: torch.Tensor, y: torch.Tensor):
    """
    计算预测正确的数量
    :param y_hat: 预测值
    :param y: 真实值
    :return: 正确的数量
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y_hat.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):
    """
    计算在指定数据集上的预测正确率
    :param net: 模型
    :param data_iter: 数据集
    :return: 正确率
    """
    if isinstance(net, torch.nn.Module):
        net.eval()  # 模型进入评估模式
    metric = Accumulator(2)  # 用来存储正确数量和总数量
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            metric.add(accuracy(y_hat, y), y.numel())
    return metric[0] / metric[1]


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """
    使用GPU计算模型在数据集上的精度
    :param net: 模型
    :param data_iter: 数据
    :param device: 设备
    :return:
    """
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def evaluate_loss(net, data_iter, loss):  # @save
    """
    评估给定数据集上模型的损失
    :param net: 模型
    :param data_iter: 数据集
    :param loss: 损失函数
    :return: 平均损失
    """
    metric = Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


class Animator:
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(9, 5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
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
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def train_epoch_ch3(net, train_iter, loss, updater):
    """
    训练模型的一个迭代周期
    :param net: 模型
    :param train_iter: 训练数据
    :param loss: 损失函数
    :param updater: 优化函数
    :return: 平均损失，正确率
    """
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)  # 训练总损失和，训练正确数总和， 样本总数
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """
    训练模型
    :param net: 模型
    :param train_iter: 训练集
    :param test_iter: 测试集
    :param loss: 损失函数
    :param num_epochs: 迭代周期数
    :param updater: 优化函数
    :return:
    """
    animator = Animator(xlabel='迭代周期', xlim=[1, num_epochs], ylim=[0.3, 0.9],  # x轴、y轴刻度上下限
                        legend=['平均损失', '训练正确率', '测试正确率'], figsize=(10.5, 6))
    train_metrics = float('inf'), 0
    test_acc = 0
    for i in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(i + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics  # 损失，训练精度
    assert train_loss < 0.5, train_loss
    assert 0.7 < train_acc < 1, train_acc
    assert 0.8 < test_acc < 1, test_acc


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('正在 {} 上训练。。。'.format(device))
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='迭代周期', xlim=[1, num_epochs], legend=['训练损失', '训练精度', '测试精度'])
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
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
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'损失 {train_l:.3f}, 训练精度 {train_acc:.3f},  测试精度 {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} 个样本/秒 在 {str(device)} 上')


def predict_ch3(net, test_iter, n=10):
    """
    预测标签
    :param net: 模型
    :param test_iter: 测试数据
    :param n: 测试数据量
    :return:
    """
    for X, y in test_iter:
        true_labels = get_fashion_mnist_labels(y)
        pred_labels = get_fashion_mnist_labels(net(X).argmax(axis=1))
        titles = [f'{p}[{t}]' for t, p in zip(true_labels, pred_labels)]
        show_images(X[:n].reshape((n, 28, 28)), 2, n // 2, titles)
        break


def try_gpu(i=0):
    """
    如果有GPU则返回，如果没有则返回CPU
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """
    返回所有的gpu列表，如果没有gpu则返回cpu
    :return:
    """
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def corr2d(X, K):
    """
    计算二维的互相关运算
    :param X: 二维张量
    :param K: 卷积核
    :return: 卷积结果
    """
    xh, xw = X.shape
    kh, kw = K.shape
    Y = torch.zeros(size=(xh - kh + 1, xw - kw + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + kh, j:j + kw] * K).sum()
    return Y


if __name__ == '__main__':
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    Y = corr2d(X, K)
    print(Y)
