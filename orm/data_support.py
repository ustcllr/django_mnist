"""
为图像识别项目提供数据支持
"""

import numpy as np

# 定义读取训练集的数量
TRAINING_SUM = 60

# 定义读取测试集的数量
RECOGNIZE_SUM = 10000


def get_training_image_ary():
    """得到图片二维数组，每一列是一个训练集"""

    # 创建一个二维数组，共TRAINING_SUM张图片
    image_ary = np.zeros(shape=(784, TRAINING_SUM), dtype='int')

    with open('mnist_data/train-images.idx3-ubyte', 'rb') as f:
        f_read = f.read()

        for i in range(TRAINING_SUM):
            # 得到每个训练集的开始位置
            begin_num = 16 + i*784

            for j in range(784):
                # 数据是纵向进行填充
                image_ary[j][i] = f_read[begin_num + j]

    return image_ary


def get_training_label_ary():
    """得到标签的1维数组，用于得出Y"""

    # 初始化数组，共有TRAINING_SUM个标签
    label_ary = np.zeros((TRAINING_SUM, ), 'int')

    with open('mnist_data/train-labels.idx1-ubyte', 'rb') as f:
        f_read = f.read()

        for i in range(8, TRAINING_SUM+8):
            label_ary[i-8] = f_read[i]

    return label_ary


def get_recognize_image_ary():
    """测试集的图像数组"""

    # 创建一个二维数组，共RECOGNIZE_SUM张图片
    image_ary = np.zeros(shape=(RECOGNIZE_SUM, 784), dtype='int')

    with open('mnist_data/t10k-images.idx3-ubyte', 'rb') as f:
        f_read = f.read()

        for j in range(RECOGNIZE_SUM):
            # 得到当前图片的开始位置
            begin_num = 16 + j*784

            for i in range(784):
                image_ary[j][i] = f_read[begin_num + i]

    return image_ary


def get_recognize_label_ary():
    """测试集的标签数组"""

    # 初始化数组，共有RECOGNIZE_SUM个标签
    label_ary = np.zeros((RECOGNIZE_SUM, ), 'int')

    with open('mnist_data/t10k-labels.idx1-ubyte', 'rb') as f:
        f_read = f.read()

        for i in range(8, RECOGNIZE_SUM+8):
            label_ary[i-8] = f_read[i]

    return label_ary
