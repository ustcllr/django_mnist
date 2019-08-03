"""
根据训练出来的特征值，计算测试集的识别率
"""

import json

import numpy as np

from data_support import get_recognize_image_ary, get_recognize_label_ary, RECOGNIZE_SUM
from train_model import L

# 再定义集合的总数
m = RECOGNIZE_SUM

# 从文件中提取模型
file = open('data/array.txt', 'r')
str1 = file.read()
list1 = json.loads(str1)
w_li = list1[0]
b_li = list1[1]
wlist = [np.array(x) for x in w_li]
blist = [np.array(x) for x in b_li]
zlist = [0] * (L+1)
alist = [0] * (L+1)

# 得到训练集和标签集
alist[0] = get_recognize_image_ary()
label_ary = get_recognize_label_ary()

file.close()

# 前向传播：input layer全部使用relu作为激活函数
for l in range(1, L+1):
    zlist[l] = np.dot(wlist[l], alist[l-1]) + blist[l]
    if l != L:
        alist[l] = np.where(zlist[l]<0, 0, zlist[l])
    else:
        alist[l] = 1 / (1 + np.exp(-zlist[l]))

# 数据分析，计算识别率
correct = 0
# 将a_L进行转置，容易计算
assist = alist[L].T
for i in range(m):
    predict_ary = assist[i]
    max_rate = np.max(predict_ary)
    num_index = int(np.argwhere(predict_ary==max_rate))
    # print(num_index, max_rate)
    # 如果最大概率达到0.5，rank矩阵对应位置的值设为1
    if max_rate >= 0.5 and label_ary[i] == num_index:
        correct += 1

# correct /= m
print('correct =', correct)
