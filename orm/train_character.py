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
    """对于手写数字数据集，根据不同的标签，训练参数w和b"""

    # 初始化特征值w和b
    w = np.ones(shape=(784, ))
    # 习惯性地设为2
    b = 2

    # 再定义集合的总数
    m = TRAINING_SUM

    # 对训练集进行若干次学习
    # 190716目前认为1000次差不多了，再多会严重影响训练时间
    for t in range(1000):

        # 进行初始化，dw同样是一个一维数组
        dw = np.zeros(shape=(784, ))
        db = 0

        # 首先，根据损失函数计算每一个样本的下降梯度
        for i in range(m):
            # 首先得到xi, yi
            # 如果取出的标签与输入标签相同，则yi=1
            x = image_ary[i]
            y = (0, 1)[label_ary[i] == label]

            # 通过逻辑回归，计算出该样本的预测值
            z = np.dot(w, x) + b
            a = 1/(1+np.e**(-z))

            # 将每个样本的下降梯度进行加权
            dw += x * (a-y)
            db += (a-y)

        # 然后，求加权平均
        dw = dw / m
        db = db / m

        # 最后，将特征值更新为新的特征值
        # 机器学习率根据经验定义为10
        lr = 10
        w = w - lr*dw
        b = b - lr*db

        # 每训练10次，打印一下次数和最新的b
        if (t+1) % 10 == 0:
            print('label = {}, t = {}, b = {}'.format(label, t+1, b))

    # 返回两个变量
    return w, b


if __name__ == '__main__':
    w, b = get_character(3)
    print(w, b)
