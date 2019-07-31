"""
使用基于成本函数的梯度下降法训练特征值
对于每个标签，都要训练大量的正确项和干扰项
特征值用于测试集的识别，期望识别率95+
"""

import numpy as np

from .data_support import get_training_image_ary, get_training_label_ary, TRAINING_SUM


# 得到训练集和标签集
a_0 = get_training_image_ary()
label_ary = get_training_label_ary()


def get_network(label):
    """输入标签，得到n_1*1神经网络的特征值"""

    # 定义训练集的总数
    m = TRAINING_SUM

    # 定义第1层节点的个数
    n_1 = 100

    # 定义神经网络的初始特征向量
    # 第一层ANN使用Relu作为激活函数，为了使梯度下降得更快，初始值均为正数
    w_1 = np.random.randn(n_1, 784) * 0.01
    # b的值无所谓，可以设为全0
    b_1 = np.zeros((n_1, 1))

    # 第二层由于使用逻辑回归函数，初始值要尽量靠近原点
    w_2 = np.random.randn(1, n_1) * 0.01
    # 由于都是向量，所以实数也统一成矩阵
    b_2 = np.zeros((1, 1))

    # 获得行向量Y
    y = np.where(label_ary==label, 1, 0)

    # 对训练集进行1000次训练，再多会影响训练时间
    for t in range(600):

        z_1 = np.dot(w_1, a_0) + b_1
        # 第一层激活函数为relu
        a_1 = np.where(z_1<0, 0, z_1)

        z_2 = np.dot(w_2, a_1) + b_2
        # 第2层激活函数为sigmoid
        a_2 = 1 / (1 + np.exp(-z_2))

        # 反向得出n_1个特征向量的增量
        dz_2 = a_2 - y
        dw_2 = np.dot(dz_2, a_1.T) / m
        db_2 = np.sum(dz_2) / m

        dg_1 = np.where(z_1<0, 0, 1)
        dz_1 = np.dot(w_2.T, dz_2) * dg_1
        dw_1 = np.dot(dz_1, a_0.T) / m
        db_1 = np.sum(dz_1, axis=1).reshape(n_1, 1) / m

        # 最后，将特征值更新为新的特征值
        # 定义机器学习率
        lr = 0.001
        w_1 = w_1 - lr * dw_1
        b_1 = b_1 - lr * db_1
        w_2 = w_2 - lr * dw_2
        b_2 = b_2 - lr * db_2

        # 每训练10次，输出并打印次数和最新距离
        if (t+1) % 10 == 0:
            line = 'label = {}, t = {}, distance = {}'.format(label, t+1, np.sum(abs(dz_2)))
            print(line)
            if (t+1) % 100 == 0:
                file = open('log.txt', 'a')
                file.write(line)
                file.write('\n')
                file.close()

    # 返回特征值列表
    return [w_1, b_1, w_2, b_2]
