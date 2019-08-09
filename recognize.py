"""
根据训练出来的特征值，计算测试集的识别率
"""

import os
import json

import numpy as np

from data_support import get_recognize_image_ary, get_recognize_label_ary, RECOGNIZE_SUM, DATASET_NAME
from utils import forward_prop, get_variance, get_bias
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
w = [np.array(x) for x in w_li]
b = [np.array(x) for x in b_li]
mu = np.array(input_dict['mu'])
sigma = np.array(input_dict['sigma'])
bias1 = np.array(input_dict['bias'])

# 得到输入层和输出层向量
x = get_recognize_image_ary()
y = get_recognize_label_ary()

# 将输入向量进行归一化
x_normal = (x - mu) / sigma

# 前向传播，得到a_L
z, a = forward_prop(w, b, x_normal, L)

# 计算偏差
bias2 = get_bias(a[L], y, m)

# 计算方差
variance = get_variance(bias1, bias2)
variance = round(variance, 6)

correct_rate = round((1 - bias2) * 100, 2)
print('correct_rate =', correct_rate)
print('variance =', variance)
