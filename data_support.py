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


def get_training_image_ary():
    """得到输入层矩阵，每一列是一个训练集"""

    # 通过数据集二进制文件创建一个二维数组，数据类型为无符号8位
    loaded = np.fromfile(file='data/train-images.idx3-ubyte', dtype=np.uint8)
    # 将数组变换为我们需要的a_0
    a_0 = loaded[16: TRAINING_SUM*784+16].reshape(TRAINING_SUM, 784).T
    return a_0


def get_training_label_ary():
    """得到输出层矩阵，每一行是一个标签的特征"""

    # 通过数据集二进制文件创建一个二维数组，数据类型为无符号8位
    loaded = np.fromfile(file='data/train-labels.idx1-ubyte', dtype=np.uint8)
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
    loaded = np.fromfile(file='data/t10k-images.idx3-ubyte', dtype=np.uint8)
    # 将数组变换为我们需要的a_0
    a_0 = loaded[16: RECOGNIZE_SUM*784+16].reshape(RECOGNIZE_SUM, 784).T
    return a_0


def get_recognize_label_ary():
    """测试集的标签数组"""

    # 通过数据集二进制文件创建一个二维数组，数据类型为无符号8位
    loaded = np.fromfile(file='data/t10k-labels.idx1-ubyte', dtype=np.uint8)
    label_ary = loaded[8: RECOGNIZE_SUM+8]
    return label_ary


if __name__ == '__main__':
    print(datetime.datetime.now())
    y = get_training_label_ary()
    print(y)
    print(datetime.datetime.now())
