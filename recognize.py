"""
根据训练出来的特征值，计算测试集的识别率
"""

import os
import json

import numpy as np

from data_support import get_training_image_ary, get_training_label_ary, \
        get_recognize_image_ary, get_recognize_label_ary, DATASET_NAME
from utils import get_normalize, forward_prop, get_variance, get_accuracy
from train_model import L


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
mu_x = np.array(input_dict['mu_x'])
sigma_x = np.array(input_dict['sigma_x'])

# 计算训练集的识别率
x = get_training_image_ary()
y = get_training_label_ary()
x_norm = (x - mu_x) / sigma_x
z, a = forward_prop(x_norm, w, b)
accuracy1 = get_accuracy(a[L], y)

# 计算测试集的识别率
x = get_recognize_image_ary()
y = get_recognize_label_ary()
x_norm = (x - mu_x) / sigma_x
z, a = forward_prop(x_norm, w, b)
accuracy2 = get_accuracy(a[L], y)

print('accuracy1 =', accuracy1)
print('accuracy2 =', accuracy2)
