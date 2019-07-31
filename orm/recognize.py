"""
根据训练出来的特征值，计算测试集的识别率
"""

import numpy as np

from .data_support import get_recognize_image_ary, get_recognize_label_ary, RECOGNIZE_SUM


# 得到训练集和标签集
recognize_image_ary = get_recognize_image_ary()
recognize_label_ary = get_recognize_label_ary()

# 再定义集合的总数
m = RECOGNIZE_SUM


def recognize_rank(chatacter_dict):
    """输入一个模型，输出一个整体的识别率"""

    correct_recognize = 0

    # 对每一个测试集进行识别
    for i in range(m):
        # 得到该测试集的向量
        x = recognize_image_ary[:,i].reshape(784, 1)

        rank_list = []
        for j in range(10):
            w_1, b_1, w_2, b_2 = [x for x in chatacter_dict[j]]

            z_1 = np.dot(w_1, x) + b_1
            # 第一层激活函数为relu
            a_1 = np.where(z_1<0, 0, z_1)

            z_2 = np.dot(w_2, a_1) + b_2
            # 第2层激活函数为sigmoid
            a_2 = 1 / (1 + np.exp(-z_2))

            rank_list.append(a_2[0][0])

        # 获得rank出来的答案
        recognize_value = rank_list.index(max(rank_list))

        # 对答案
        if (recognize_label_ary[0][i] == recognize_value) and max(rank_list) >= 0.5:
            correct_recognize += 1

    # 计算识别率
    correct_recognize_rate = round(correct_recognize/m, 2)
    return correct_recognize_rate


def recognize_single(label, w, b):
    """单边识别率的计算，返回正确项和干扰项的识别率"""

    # 这一组变量用于求正确率和错误率
    correct_total = 0
    correct_recognize = 0
    wrong_total = 0
    wrong_recognize = 0

    # 对每一个测试集进行识别
    for i in range(m):
        # 得到该测试集的向量
        x = recognize_image_ary[i]
        # 根据罗辑回归函数计算预测值
        z = np.dot(w, x) + b
        a = 1/(1+np.e**(-z))

        # 数据统计
        if recognize_label_ary[i] == label:
            # 正确项总数++
            correct_total += 1
            # 如果识别对了，正确项识别++
            if a >= 0.5:
                correct_recognize += 1
        else:
            # 干扰项总数++
            wrong_total += 1
            # 如果识别对了，干扰项识别++
            if a < 0.5:
                wrong_recognize += 1

    # 计算识别率
    correct_recognize_rate = round(correct_recognize/correct_total, 2)
    wrong_recognize_rate = round(wrong_recognize/wrong_total, 2)
    return correct_recognize_rate, wrong_recognize_rate
