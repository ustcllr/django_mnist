"""
为图像识别项目提供数据支持
"""

import os
import datetime
import numpy as np

# 定义读取训练集的数量
TRAINING_SUM = 60000

# 定义读取测试集的数量
RECOGNIZE_SUM = 10000

# 定义数据集的名称
DATASET_NAME = 'mnist'


def get_training_image_ary():
    """得到输入层矩阵，每一列是一个训练集"""

    # 通过数据集二进制文件创建一个二维数组，数据类型为无符号8位
    file_name = os.path.join(DATASET_NAME, 'train-images.idx3-ubyte')
    loaded = np.fromfile(file=file_name, dtype=np.uint8)
    # 将数组变换为我们需要的a_0
    a_0 = loaded[16: TRAINING_SUM*784+16].reshape(TRAINING_SUM, 784).T
    return a_0


def get_training_label_ary():
    """得到输出层矩阵，每一行是一个标签的特征"""

    # 通过数据集二进制文件创建一个二维数组，数据类型为无符号8位
    file_name = os.path.join(DATASET_NAME, 'train-labels.idx1-ubyte')
    loaded = np.fromfile(file=file_name, dtype=np.uint8)
    y_ori = loaded[8: TRAINING_SUM+8]
    # 先初始化一个10行矩阵
    y = np.zeros((10, TRAINING_SUM), 'int')
    # 再进行二分
    for i in range(10):
        y[i] = np.where(y_ori==i, 1, 0)
    return y


def get_recognize_image_ary():
    """测试集的图像数组"""

    # 通过数据集二进制文件创建一个二维数组，数据类型为无符号8位
    file_name = os.path.join(DATASET_NAME, 't10k-images.idx3-ubyte')
    loaded = np.fromfile(file=file_name, dtype=np.uint8)
    # 将数组变换为我们需要的a_0
    a_0 = loaded[16: RECOGNIZE_SUM*784+16].reshape(RECOGNIZE_SUM, 784).T
    return a_0


def get_recognize_label_ary():
    """测试集的标签数组"""

    # 通过数据集二进制文件创建一个二维数组，数据类型为无符号8位
    file_name = os.path.join(DATASET_NAME, 't10k-labels.idx1-ubyte')
    loaded = np.fromfile(file=file_name, dtype=np.uint8)
    y_ori = loaded[8: RECOGNIZE_SUM+8]
    # 先初始化一个10行矩阵
    y = np.zeros((10, RECOGNIZE_SUM), 'int')
    # 再进行二分
    for i in range(10):
        y[i] = np.where(y_ori==i, 1, 0)
    return y


def forward_prop(w, b, x, L):
    """前向传播，输入特征列表，第0层向量，最大层数。输出预测列表a和暂存的z"""

    # 列表初始化
    a = [np.array([0])] * (L+1)
    z = [np.array([0])] * (L+1)
    a[0] = x

    # 目前规定，隐藏层全部使用RELU，输出层使用逻辑回归
    for l in range(1, L+1):
        z[l] = np.dot(w[l], a[l-1]) + b[l]
        if l != L:
            a[l] = np.where(z[l]<0, 0, z[l])
        else:
            # 190805改良逻辑回归，利用中心对称，防止出现得到INF的情况。虽然不影响，但是不严谨。
            a[l] = np.where(z[l]>0, 1 / (1 + np.exp(-z[l])), np.exp(z[l]) / (1 + np.exp(z[l])))

    return z, a


def back_prop(w, z, a, y, L, m):
    """后向传播，输入特征列表w，预测值a和暂存值z，输出矩阵y。输出特征向量的增量"""

    # 初始化中间增量
    dz = [np.array([0])] * (L+1)
    dg = [np.array([0])] * (L+1)
    # 初始化输出增量
    dw = [np.array([0])] * (L+1)
    db = [np.array([0])] * (L+1)

    # 后向传播，第L层根据逻辑回归可以直接给出
    for l in range(L, 0, -1):
        if l == L:
            dz[l] = a[l] - y
        else:
            dg[l] = np.where(z[l]<0, 0, 1)
            dz[l] = np.dot(w[l+1].T, dz[l+1]) * dg[l]
        dw[l] = np.dot(dz[l], a[l-1].T) / m
        db[l] = np.sum(dz[l], axis=1, keepdims=True) / m

    return dw, db


def get_variance(a_L, y, m):
    """计算偏差，根据预测值和实际值，返回偏差率的列表"""

    # 创建用于对比的辅助矩阵
    assist = np.where(a_L>=0.5, 1, 0)
    variance = np.sum(abs(assist - y), axis=1) / m
    return variance.tolist()


def get_bias(v1, v2):
    """根据两个偏差列表，计算方差"""

    base_sum = 0
    for i in range(len(v1)):
        base_sum += (v1[i] - v2[i]) ** 2
    bias = np.sqrt(base_sum)
    return bias


if __name__ == '__main__':
    y = get_recognize_label_ary()
    print(y.shape)
