# -*- coding: utf-8 -*-
"""
@File    : utils.py
@Time    : 2024/3/19 15:11
@Desc    : 
"""
import hashlib
import os
import tarfile
import zipfile
import requests
import torchvision
from torch.utils import data
from torchvision import transforms

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
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):  # @save
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


def download_all():  # @save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)


DATA_HUB['kaggle_house_train'] = (DATA_URL + 'kaggle_house_pred_train.csv',
                                  '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (DATA_URL + 'kaggle_house_pred_test.csv',
                                 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')


def load_data_cifar10(batch_size, resize=None):
    """
    加载加载CIFAR-10数据集
    :param batch_size: 批量块大小
    :param resize:
    :return:
    """
    data_transform = [transforms.ToTensor()]
    if resize:
        data_transform.insert(0, transforms.Resize(resize))  # 将图像调整为指定大小，以便与模型的输入尺寸匹配
    trans = transforms.Compose(data_transform)
    train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=trans)
    test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=trans)
    train_data = data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_data = data.DataLoader(test_dataset, batch_size, shuffle=False)
    return train_data, test_data


def get_zh_label(label):
    label_to_class = {
        0: '飞机', 1: '汽车', 2: '鸟类', 3: '猫', 4: '鹿', 5: '狗', 6: '青蛙', 7: '马', 8: '船', 9: '卡车'
    }
    return label_to_class[label]


if __name__ == '__main__':
    train_iter, test_iter = load_data_cifar10(32)
    for X, y in train_iter:
        print(X)
        print(y)
        break
