"""
训练神经网络的模型
"""

import os
import datetime
import json

import numpy as np

from data_support import get_training_image_ary, get_training_label_ary, DATASET_NAME
from utils import get_normalize, forward_prop, back_prop, get_accuracy


# 得到输入层和输出层向量
x = get_training_image_ary()
y = get_training_label_ary()

# 定义训练集的总数
m = 60000

# 定义节点数列表
n = [x.shape[0], 100, 50, 10]

# 定义层的总数
L = len(n) - 1

# 定义循环次数
times = 30

# 定义机器学习率
lr = 0.5

# 定义mini-batch大小
mini_size = 1000

# 得到batch数目
batch_num = int(m / mini_size)


def training():
    """训练模型，得到特征向量，超参数向量和偏差"""

    # 将输入向量进行归一化
    mu_x, sigma_x, x_norm = get_normalize(x)

    # 初始化特征向量
    w = [np.zeros(1)] * (L+1)
    b = [np.zeros(1)] * (L+1)

    # 对于RELU，尽量控制输出层的方差也为1，对于逻辑回归，都设在原点附近
    for l in range(1, L+1):
        if l < L:
            # 初始化权重
            ir = np.sqrt(2 / n[l-1])
        else:
            ir = 0.01
        w[l] = np.random.randn(n[l], n[l-1]) * ir
        b[l] = np.zeros([n[l], 1])

    # 循环训练模型，如果学习率够快，50次就够了
    for t in range(times):

        # 使用mini-batch，加快遍历速度，并减少遍历次数
        for seq in range(batch_num):
            x_mini = x_norm.T[seq*mini_size: (seq+1)*mini_size].T
            y_mini = y.T[seq*mini_size: (seq+1)*mini_size].T

            # 进行一次前向传播
            z, a = forward_prop(x_mini, w, b)

            # 进行一次后向传播
            dw, db = back_prop(y_mini, a, z, w)

            # 两次传播之后，更新需要学习的向量
            for l in range(1, L+1):
                w[l] = w[l] - lr * dw[l]
                b[l] = b[l] - lr * db[l]

            # 每训练10次，计算成本函数，观察梯度是否下降
            if (t+1) % 10 == 0 and seq == 20:
                j = np.sum(- y_mini * np.log(a[L])).astype(np.int32)
                accuracy = get_accuracy(a[L], y_mini)
                print('t = {}, j = {}, accuracy = {}'.format(t+1, j, accuracy))

    return w, b, mu_x, sigma_x


if __name__ == '__main__':
    print('begin_time =', datetime.datetime.now())
    w, b, mu_x, sigma_x = training()
    print('end_time =', datetime.datetime.now())

    # 创建一个字典
    w_li = [x.tolist() for x in w]
    b_li = [x.tolist() for x in b]
    output_dict = {
        'w_li': w_li,
        'b_li': b_li,
        'mu_x': mu_x.tolist(),
        'sigma_x': sigma_x.tolist(),
    }

    # 转成JSON字符串后存入文件
    output_json = json.dumps(output_dict)
    write_path = os.path.join(DATASET_NAME, 'array.txt')
    file = open(write_path, 'w')
    file.write(output_json)
    file.close()
