"""
提供各种计算工具
"""

import numpy as np


def get_normalize(x, m):
    """根据输入向量，计算归一化参数mu和sigma"""

    mu = np.sum(x, axis=1, keepdims=True) / m
    # 创建辅助矩阵，将输入向量中心化，便于计算方差归一常数
    assist = x - mu
    # 计算方差归一常数
    square_sum = np.sum(assist**2, axis=1, keepdims=True)
    sigma = np.sqrt(square_sum / m)
    # 如果算出是0，说明方差不用改变，改成1
    sigma = np.where(sigma==0, 1, sigma)
    return mu, sigma


def forward_prop(w, b, x, L):
    """前向传播，输入特征向量和x，输出z和a"""

    # 列表初始化
    z = [np.array([0])] * (L+1)
    a = [np.array([0])] * (L+1)
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


def back_prop(w, z, a, y, L, m):
    """后向传播，输入特征向量w，预测值z和a，输出矩阵y。输出特征向量的增量"""

    # 初始化中间增量
    dz = [np.array([0])] * (L+1)
    dg = [np.array([0])] * (L+1)
    # 初始化输出增量
    dw = [np.array([0])] * (L+1)
    db = [np.array([0])] * (L+1)

    # 后向传播，第L层直接给出，前面使用递归
    for l in range(L, 0, -1):
        if l == L:
            dz[l] = a[l] - y
        else:
            dg[l] = np.where(z[l]<0, 0, 1)
            dz[l] = np.dot(w[l+1].T, dz[l+1]) * dg[l]
        dw[l] = np.dot(dz[l], a[l-1].T) / m
        db[l] = np.sum(dz[l], axis=1, keepdims=True) / m

    return dw, db


def get_bias(a_L, y, m):
    """根据预测值和实际值，计算当前数据集的偏差"""

    # 通过循环计算分类正确的数量
    correct = 0
    for i in range(m):
        # 使用Hardmax处理预测值
        a_L.T[i] = np.where(a_L.T[i]==np.max(a_L.T[i]), 1, 0)
        # 如果处理过的预测向量与标签向量一致，说明识别正确
        if (y.T[i] == a_L.T[i]).all():
            correct += 1

    # 计算偏差
    bias = (m - correct) / m
    return bias


def get_variance(b1, b2):
    """根据训练集和测试集的偏差，计算方差"""

    variance_sqa = (b1 - b2) ** 2
    variance = np.sqrt(variance_sqa)
    return variance
