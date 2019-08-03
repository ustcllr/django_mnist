"""
训练神经网络的模型
"""

import os
import datetime
import json

import numpy as np

from data_support import get_training_image_ary, get_training_label_ary, \
        TRAINING_SUM, DATASET_NAME


# 定义训练集的总数
m = TRAINING_SUM

# 定义节点数列表
nlist = [784, 100, 50, 10]

# 定义层的总数
L = len(nlist) - 1

# 定义初始化常数
c_ini = 0.01

# 定义循环次数
times = 1000

# 定义机器学习率
c_lr = 0.01

def training():
    """训练模型，根据超参数，返回两个特征值列表"""

    # 定义每一层都可能需要用到的列表gzwb
    wlist = [0] * (L+1)
    blist = [0] * (L+1)
    zlist = [0] * (L+1)
    alist = [0] * (L+1)
    dglist = [0] * (L+1)
    dzlist = [0] * (L+1)
    dwlist = [0] * (L+1)
    dblist = [0] * (L+1)

    # 特征向量初始化，都设在原点附近
    for l in range(0, L+1):
        # 第0层不需要特征向量，设为0
        if l == 0:
            wlist[l] = np.array([0])
            blist[l] = np.array([0])
        else:
            wlist[l] = np.random.randn(nlist[l], nlist[l-1]) * c_ini
            blist[l] = np.zeros((nlist[l], 1))

    # 得到训练集和标签集
    alist[0] = get_training_image_ary()
    y = get_training_label_ary()

    # 循环训练模型，一般为1000次
    for t in range(times):

        # 前向传播：input layer全部使用relu作为激活函数
        for l in range(1, L+1):
            zlist[l] = np.dot(wlist[l], alist[l-1]) + blist[l]
            if l != L:
                alist[l] = np.where(zlist[l]<0, 0, zlist[l])
            else:
                alist[l] = 1 / (1 + np.exp(-zlist[l]))

        # 反向传播
        for l in range(L, 0, -1):
            if l == L:
                dzlist[l] = alist[l] - y
            else:
                dglist[l] = np.where(zlist[l]<0, 0, 1)
                dzlist[l] = np.dot(wlist[l+1].T, dzlist[l+1]) * dglist[l]
            dwlist[l] = np.dot(dzlist[l], alist[l-1].T) / m
            dblist[l] = np.sum(dzlist[l], axis=1, keepdims=True) / m

        # 最后，将特征值更新为新的特征值
        for l in range(1, L+1):
            wlist[l] = wlist[l] - c_lr * dwlist[l]
            blist[l] = blist[l] - c_lr * dblist[l]

        # 每训练10次，打印次数和最新偏差
        if (t+1) % 10 == 0:
            # 定义用来计算偏差的辅助矩阵
            # print(blist[L])
            assist = np.where(alist[L]>=0.5, 1, 0)
            variance = np.sum(abs(assist - y), axis=1).tolist()
            print('t = {}, variance = {}'.format(t+1, variance))

    return wlist, blist


if __name__ == '__main__':
    print('begin_time =', datetime.datetime.now())
    wlist, blist = training()
    print('end_time =', datetime.datetime.now())

    # 存入一个文件
    w_li = [x.tolist() for x in wlist]
    b_li = [x.tolist() for x in blist]
    list1 = [w_li, b_li]
    str1 = json.dumps(list1)
    write_path = os.path.join(DATASET_NAME, 'array.txt')
    file = open(write_path, 'w')
    file.write(str1)
    file.close()
