"""
根据训练出来的特征值，计算测试集的识别率
"""

import os
import json

import numpy as np

from data_support import get_recognize_image_ary, get_recognize_label_ary, \
        forward_prop, get_variance, get_bias, \
        RECOGNIZE_SUM, DATASET_NAME
from train_model import L


# 定义集合的总数
m = RECOGNIZE_SUM

# 读取文件
read_path = os.path.join(DATASET_NAME, 'array.txt')
file = open(read_path, 'r')
input_dict = json.loads(file.read())
file.close()

# 从文件中提取模型
w_li = input_dict['w_li']
b_li = input_dict['b_li']
v1 = input_dict['variance']
w = [np.array(x) for x in w_li]
b = [np.array(x) for x in b_li]

# 得到训练集和标签集
x = get_recognize_image_ary()
y = get_recognize_label_ary()

# 前向传播，得到a_L
z, a = forward_prop(w, b, x, L)

# 创建辅助矩阵，用于计算识别率
alt = a[L].T
yt = y.T
assist = np.zeros(alt.shape)
for i in range(m):
    predict_ary = alt[i]
    max_rate = np.max(predict_ary)
    num_index = int(np.argwhere(predict_ary==max_rate))
    # 如果最大概率达到0.5，辅助矩阵对应位置的值设为1
    if max_rate >= 0.5:
        assist[i][num_index] = 1

# 计算识别率
correct = 0
for i in range(m):
    if (assist[i] == yt[i]).all():
        correct += 1
correct = correct / m * 100

# 计算方差
v2 = get_variance(a[L], y, m)
bias = get_bias(v1, v2)

print('correct_rate =', correct)
print('bias =', bias)
