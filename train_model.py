"""
训练神经网络的模型
"""

import os
import datetime
import json

import numpy as np

from data_support import get_training_image_ary, get_training_label_ary, TRAINING_SUM, DATASET_NAME
from utils import forward_prop, back_prop, get_normalize, get_bias


# 得到输入层和输出层向量
x = get_training_image_ary()
y = get_training_label_ary()

# 定义训练集的总数
m = TRAINING_SUM

# 定义节点数列表
n = [x.shape[0], 100, 50, 10]

# 定义层的总数
L = len(n) - 1

# 定义循环次数
times = 400

# 定义机器学习率
lr = 1


def training():
    """训练模型，得到特征向量，超参数向量和偏差"""

    # 将输入向量进行归一化
    mu, sigma = get_normalize(x, m)
    x_normal = (x - mu) / sigma

    # 初始化特征向量
    w = [np.array([0])] * (L+1)
    b = [np.array([0])] * (L+1)

    # 对于RELU，尽量控制输出层的方差也为1，对于逻辑回归，都设在原点附近
    for l in range(1, L+1):
        # 对于RELU，第l层的初始化常数与第l-1层的节点数相关
        if l < L:
            ir = np.sqrt(2 / n[l-1])
        else:
            ir = 0.01
        w[l] = np.random.randn(n[l], n[l-1]) * ir
        b[l] = np.zeros((n[l], 1))

    # 循环训练模型，如果学习率够快，200次就够了
    for t in range(times):

        # 进行一次前向传播
        z, a = forward_prop(w, b, x_normal, L)

        # 进行一次后向传播
        dw, db = back_prop(w, z, a, y, L, m)

        # 两次传播之后，更新特征向量
        for l in range(1, L+1):
            w[l] = w[l] - lr * dw[l]
            b[l] = b[l] - lr * db[l]

        # 每训练10次，计算成本函数，观察梯度是否下降
        if (t+1) % 10 == 0:
            # # 计算逻辑回归的输入层前20个特征的方差，看看是否归一
            # square_sum = np.sum(a[L-1]**2, axis=1)
            # variance = square_sum / m
            # print('variance =', variance[20: 30])

            j = np.sum(- y * np.log(a[L])).astype(int)
            bias = get_bias(a[L], y, m)
            correct_rate = round((1 - bias) * 100, 2)
            print('t = {}, j = {}, correct_rate = {}'.format(t+1, j, correct_rate))

    # 根据训练到最后一次得到的a_L，计算偏差
    bias = get_bias(a[L], y, m)
    return w, b, mu, sigma, bias


if __name__ == '__main__':
    # training()
    print('begin_time =', datetime.datetime.now())
    w, b, mu, sigma, bias = training()
    print('end_time =', datetime.datetime.now())

    # 创建一个字典
    w_li = [x.tolist() for x in w]
    b_li = [x.tolist() for x in b]
    output_dict = {
        'w_li': w_li,
        'b_li': b_li,
        'mu': mu.tolist(),
        'sigma': sigma.tolist(),
        'bias': bias,
    }

    # 转成JSON字符串后存入文件
    output_json = json.dumps(output_dict)
    write_path = os.path.join(DATASET_NAME, 'array.txt')
    file = open(write_path, 'w')
    file.write(output_json)
    file.close()
