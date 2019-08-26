"""
为图像识别项目提供数据支持
入乡随俗，对于TF数据集就竖着排
"""

import os
import numpy as np

# 定义数据集的名称
DATASET_NAME = 'mnist'


def get_training_image_ary():
    """得到输入层矩阵，每一列是一个训练集"""

    # 通过数据集二进制文件创建一个二维数组，数据类型为无符号8位
    file_name = os.path.join(DATASET_NAME, 'train-images.idx3-ubyte')
    loaded = np.fromfile(file=file_name, dtype=np.uint8)
    # 将数组变换为我们需要的a_0
    a_0 = loaded[16: ].reshape(60000, 784)
    return a_0


def get_training_label_ary():
    """得到输出层矩阵，每一行是一个标签的特征"""

    # 通过数据集二进制文件创建一个二维数组，数据类型为无符号8位
    file_name = os.path.join(DATASET_NAME, 'train-labels.idx1-ubyte')
    loaded = np.fromfile(file=file_name, dtype=np.uint8)
    y_ori = loaded[8: ]
    # 先初始化矩阵
    y = np.zeros([60000, 10])
    # 再进行打孔
    for i in range(60000):
        y[i][y_ori[i]] = 1
    return y


def get_recognize_image_ary():
    """测试集的图像数组"""

    # 通过数据集二进制文件创建一个二维数组，数据类型为无符号8位
    file_name = os.path.join(DATASET_NAME, 't10k-images.idx3-ubyte')
    loaded = np.fromfile(file=file_name, dtype=np.uint8)
    # 将数组变换为我们需要的a_0
    a_0 = loaded[16: ].reshape(10000, 784)
    return a_0


def get_recognize_label_ary():
    """测试集的标签数组"""

    # 通过数据集二进制文件创建一个二维数组，数据类型为无符号8位
    file_name = os.path.join(DATASET_NAME, 't10k-labels.idx1-ubyte')
    loaded = np.fromfile(file=file_name, dtype=np.uint8)
    y_ori = loaded[8: ]
    # 先初始化矩阵
    y = np.zeros([10000, 10])
    # 再进行打孔
    for i in range(10000):
        y[i][y_ori[i]] = 1
    return y


if __name__ == '__main__':
    y = get_training_image_ary()
    print(y[100])
    # print(y.dtype)
