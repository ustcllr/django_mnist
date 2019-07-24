"""
使用基于成本函数的梯度下降法训练特征值
对于每个标签，都要训练大量的正确项和干扰项
特征值用于测试集的识别，期望识别率95+
"""

import numpy as np

from .data_support import get_training_image_ary, get_training_label_ary, TRAINING_SUM


# 得到训练集和标签集
image_ary = get_training_image_ary()
label_ary = get_training_label_ary()


def get_character(label):
    """输入标签，输出特征值w和b"""

    # 使用随机整数初始化特征向量
    w = np.random.randint(-1000, 1001, [784, 1])
    # 习惯性地设为2
    b = 2

    # 再定义集合的总数
    m = TRAINING_SUM

    # 获得行向量Y
    y = np.zeros(shape=(1, m), dtype='int')
    for i in range(m):
        y[0][i] = (0, 1)[label_ary[i] == label]

    # 对训练集进行1000次训练，再多会影响训练时间
    for t in range(1000):

        # 得到一个行向量，为所有样本集的预测值
        z = np.dot(w.T, image_ary) + b
        a = 1 / (1 + np.exp(-z))

        # 得到一个行向量，为所有样本集的dz
        multi = a - y

        # 这是最有想象力的一步，利用矩阵相乘的规律计算平均值
        dw = np.dot(image_ary, multi.T) / m
        db = np.sum(multi) / m

        # 最后，将特征值更新为新的特征值
        # 定义机器学习率
        lr = 89
        w = w - lr*dw
        b = b - lr*db

        # 每训练10次，打印一下次数和最新的b
        if (t+1) % 10 == 0:
            print('label = {}, t = {}, b = {}'.format(label, t+1, b))

    # 返回两个变量
    return w, b
