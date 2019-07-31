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
    """得到图片二维数组，每一列是一个训练集"""

    # 直接调用使用相对路径，相对调用使用绝对路径
    path = os.path.join(os.path.dirname(__file__), '../mnist_data/train-images.idx3-ubyte')
    # 通过数据集二进制文件创建一个二维数组，数据类型为无符号8位
    loaded = np.fromfile(file=path, dtype=np.uint8)
    # 将数组变换为我们需要的a_0
    a_0 = loaded[16:].reshape(60000, 784).T
    return a_0


def get_training_label_ary():
    """得到标签的1维数组，用于得出Y"""

    # 直接调用使用相对路径，相对调用使用绝对路径
    path = os.path.join(os.path.dirname(__file__), '../mnist_data/train-labels.idx1-ubyte')
    # 通过数据集二进制文件创建一个二维数组，数据类型为无符号8位
    loaded = np.fromfile(file=path, dtype=np.uint8)
    # 将数组变换为我们需要的a_0
    label_ary = loaded[8:].reshape(1, 60000)
    return label_ary


def get_recognize_image_ary():
    """测试集的图像数组"""

    # 直接调用使用相对路径，相对调用使用绝对路径
    path = os.path.join(os.path.dirname(__file__), '../mnist_data/t10k-images.idx3-ubyte')
    # 通过数据集二进制文件创建一个二维数组，数据类型为无符号8位
    loaded = np.fromfile(file=path, dtype=np.uint8)
    # 将数组变换为我们需要的a_0
    a_0 = loaded[16:].reshape(10000, 784).T
    return a_0


def get_recognize_label_ary():
    """测试集的标签数组"""

    # 直接调用使用相对路径，相对调用使用绝对路径
    path = os.path.join(os.path.dirname(__file__), '../mnist_data/t10k-labels.idx1-ubyte')
    # 通过数据集二进制文件创建一个二维数组，数据类型为无符号8位
    loaded = np.fromfile(file=path, dtype=np.uint8)
    # 将数组变换为我们需要的a_0
    label_ary = loaded[8:].reshape(1, 10000)
    return label_ary


if __name__ == '__main__':
    print(datetime.datetime.now())
    a = get_training_label_ary()
    print(a)
    print(datetime.datetime.now())
