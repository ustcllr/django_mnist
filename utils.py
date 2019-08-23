"""
提供各种计算工具
"""

import numpy as np


def get_normalize(x):
    """获得归一化后的向量以及mu和sigma参数"""

    m = x.shape[1]
    mu = np.sum(x, axis=1, keepdims=True) / m
    # 将向量中心化
    x_norm = x - mu
    # 计算方差归一常数
    sigma_square = np.sum(x_norm**2, axis=1, keepdims=True) / m
    sigma = np.sqrt(sigma_square + 0.001)
    x_norm = x_norm / sigma
    return mu, sigma, x_norm


def forward_prop(x, w, b):
    """正向传播，输入x和特征向量，输出z和a"""

    # 获得最高层数
    L = len(w) - 1

    # 初始化向量
    z = [np.zeros(1)] * (L+1)
    a = [np.zeros(1)] * (L+1)
    a[0] = x

    # 前向传播：前面L-1层使用RELU，第L层使用Softmax
    for l in range(1, L+1):
        z[l] = np.dot(w[l], a[l-1]) + b[l]
        if l < L:
            a[l] = np.where(z[l]<0, 0, z[l])
        else:
            assist = np.sum(np.exp(z[l]), axis=0)
            a[l] = np.exp(z[l]) / assist

    return z, a


def back_prop(y, a, z, w):
    """反向传播，输入y，预测值和特征向量，输出特征向量的增量"""

    # 获得当前batch的数据集数
    m = y.shape[1]
    # 获得最高层数
    L = len(w) - 1

    # 初始化向量
    dg = [np.zeros(1)] * (L+1)
    dz = [np.zeros(1)] * (L+1)
    dw = [np.zeros(1)] * (L+1)
    db = [np.zeros(1)] * (L+1)

    # 反向传播，第L层直接给出，前面使用递归
    for l in range(L, 0, -1):
        if l == L:
            dz[l] = a[l] - y
        else:
            dg[l] = np.where(z[l]<0, 0, 1)
            dz[l] = np.dot(w[l+1].T, dz[l+1]) * dg[l]
        dw[l] = np.dot(dz[l], a[l-1].T) / m
        db[l] = np.sum(dz[l], axis=1, keepdims=True) / m

    return dw, db


def get_accuracy(a_L, y):
    """输入预测值和标签，计算正答率"""

    # 得到数据集数
    m = y.shape[1]

    # 对比两个最大值一维数组
    equal_ary = np.equal(np.argmax(a_L, 0), np.argmax(y, 0))
    # 得到平均值，会将输入数组自动转换为01
    accuracy = np.mean(equal_ary)
    return round(accuracy, 3)


def get_variance(b1, b2):
    """根据训练集和测试集的偏差，计算方差"""

    variance_sqa = (b1 - b2) ** 2
    variance = np.sqrt(variance_sqa)
    return variance
