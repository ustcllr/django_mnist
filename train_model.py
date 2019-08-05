"""
训练神经网络的模型
"""

import os
import datetime
import json

import numpy as np

from data_support import get_training_image_ary, get_training_label_ary, \
        forward_prop, back_prop, get_variance, \
        TRAINING_SUM, DATASET_NAME


# 定义训练集的总数
m = TRAINING_SUM

# 定义节点数列表
n = [784, 100, 50, 10]

# 定义层的总数
L = len(n) - 1

# 定义初始化常数
ir = 0.01

# 定义循环次数
times = 1000

# 定义机器学习率
lr = 0.01


def training():
    """训练模型，根据超参数，返回两个特征值列表"""

    # 得到训练集和标签集
    x = get_training_image_ary()
    y = get_training_label_ary()

    # 初始化特征向量，都设在原点附近
    w = [np.array([0])] * (L+1)
    b = [np.array([0])] * (L+1)
    for l in range(1, L+1):
        w[l] = np.random.randn(n[l], n[l-1]) * ir
        b[l] = np.zeros((n[l], 1))

    # 循环训练模型，一般为1000次
    for t in range(times):

        # 进行一次前向传播
        z, a = forward_prop(w, b, x, L)

        # 进行一次后向传播
        dw, db = back_prop(w, z, a, y, L, m)

        # 两次传播之后，更新特征值列表
        for l in range(1, L+1):
            w[l] = w[l] - lr * dw[l]
            b[l] = b[l] - lr * db[l]

        # 每训练10次，计算成本函数，观察梯度是否下降
        if (t+1) % 50 == 0:
            j_mat = - y * np.log(a[L]) - (1-y) * np.log(1-a[L])
            # 为了观察方便，这里不再除以m
            mj = np.sum(j_mat, axis=1).astype(int).tolist()
            print('t = {}, mj = {}'.format(t+1, mj))

    # 根据训练到最后一次得到的a_L，计算偏差
    variance = get_variance(a[L], y, m)
    return w, b, variance


if __name__ == '__main__':
    print('begin_time =', datetime.datetime.now())
    w, b, variance = training()
    print('end_time =', datetime.datetime.now())

    # 创建一个字典
    w_li = [x.tolist() for x in w]
    b_li = [x.tolist() for x in b]
    output_dict = {
        'w_li': w_li,
        'b_li': b_li,
        'variance': variance,
    }

    # 转成JSON字符串后存入文件
    output_json = json.dumps(output_dict)
    write_path = os.path.join(DATASET_NAME, 'array.txt')
    file = open(write_path, 'w')
    file.write(output_json)
    file.close()
