"""
使用卷积神经网络，将识别率提升到99%
"""

import os
import datetime
import json

import numpy as np
import tensorflow as tf

from data_support import get_training_image_ary, get_training_label_ary, \
        get_recognize_image_ary, get_recognize_label_ary


# 得到训练集和测试集
x_train = tf.cast(get_training_image_ary(), tf.float32)
y_train = get_training_label_ary()
x_test = tf.cast(get_recognize_image_ary(), tf.float32)
y_test = get_recognize_label_ary()

# 获得训练集的标准化参数
# 注：一定要输入一个张量，不可以是一个数组
mu_x, sigma_x = tf.nn.moments(x_train, [0])

# 将训练集和测试集输入向量标准化
x_norm_train = tf.nn.batch_normalization(x_train, mu_x, sigma_x, 0, 1, 0.001)
x_norm_test = tf.nn.batch_normalization(x_test, mu_x, sigma_x, 0, 1, 0.001)

# 定义循环次数
times = 20

# 定义机器学习率
lr = 0.3

# 定义mini-batch大小
mini_size = 600

# 得到batch数目
batch_num = int(60000 / mini_size)


x_tf = tf.placeholder(tf.float32, [None, 784])
y_tf = tf.placeholder(tf.float32, [None, 10])

# 将输入向量从2维变成4维
x_input = tf.reshape(x_tf, [-1, 28, 28, 1])

# 根据TF的规则，过滤器按照这个维度顺序初始化
w1 = tf.Variable(tf.truncated_normal([3, 3, 1, 6], stddev=0.1))

# 根据规则，b只设定一个维度
b1 = tf.Variable(tf.constant(0.1, shape=[6, ]))

# 得到第1个卷积层
z1 = tf.nn.conv2d(x_input, w1, 1, 'SAME') + b1
a1 = tf.nn.relu(z1)

# 得到第1个最大池化层
max_pool1 = tf.nn.max_pool2d(a1, [2, 2], 2, 'SAME')

# 初始化第2个卷积层权重
w2 = tf.Variable(tf.truncated_normal([5, 5, 6, 12], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[12, ]))

# 得到第2个卷积层
z2 = tf.nn.conv2d(max_pool1, w2, 1, 'SAME') + b2
a2 = tf.nn.relu(z2)
max_pool2 = tf.nn.max_pool2d(a2, [2, 2], 2, 'SAME')

# 展开成全连接节点
a_fc0 = tf.reshape(max_pool2, [-1, 7*7*12])

# 第1个FC层
w_fc1 = tf.Variable(tf.truncated_normal([7*7*12, 10], stddev=0.1))
b_fc1 = tf.Variable(tf.zeros([10]))
z_fc1 = tf.matmul(a_fc0, w_fc1) + b_fc1
a_fc1 = tf.nn.softmax(z_fc1)

# # 第2个FC层，进行分类
# w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
# b_fc2 = tf.Variable(tf.zeros([10]))
# z_fc2 = tf.matmul(a_fc1, w_fc2) + b_fc2
# a_fc2 = tf.nn.softmax(z_fc2)

# 预防出现log0的问题，这在训练后期经常发生
a_clip = tf.clip_by_value(a_fc1, 1e-10, 1.0)
cost = tf.reduce_sum(- y_tf * tf.math.log(a_clip)) / mini_size
train = tf.compat.v1.train.GradientDescentOptimizer(lr).minimize(cost)

# 根据xy，计算模型的准确率
correct_prediction = tf.equal(tf.argmax(a_fc1, 1), tf.argmax(y_tf, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, np.float32))

# 开始训练模型
init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as session:
    # 激活TF变量
    session.run(init)
    # 打印分隔线
    print('\n{}'.format('-'*70))
    print('begin_time =', datetime.datetime.now())

    # 将训练集和测试集输入向量标准化
    x_norm_train = session.run(x_norm_train)
    x_norm_test = session.run(x_norm_test)

    for t in range(times):
        for i in range(batch_num):
            x_mini = x_norm_train[i*mini_size: (i+1)*mini_size]
            y_mini = y_train[i*mini_size: (i+1)*mini_size]

            session.run(train, feed_dict={x_tf: x_mini, y_tf: y_mini})

            # 每训练10次，计算成本函数，观察梯度是否下降
            if (t+1) % 1 == 0 and i == 94:
                cost_ran = session.run(cost, feed_dict={x_tf: x_mini, y_tf: y_mini})
                accuracy_ran = session.run(accuracy, feed_dict={x_tf: x_mini, y_tf: y_mini})
                print('t = {}, j = {}, accuracy = {}'.format(t+1, cost_ran, accuracy_ran))

    # 看一看模型的准确率
    # accuracy2 = 0
    # for i in range(100):
    #     x_mini = x_norm_test[i*mini_size: (i+1)*mini_size]
    #     y_mini = y_test[i*mini_size: (i+1)*mini_size]
    #     accuracy2 += session.run(accuracy, feed_dict={x_tf: x_mini, y_tf: y_mini})
    # accuracy2 = accuracy2 / 100
    accuracy1 = session.run(accuracy, feed_dict={x_tf: x_norm_train, y_tf: y_train})
    accuracy2 = session.run(accuracy, feed_dict={x_tf: x_norm_test, y_tf: y_test})
    print('accuracy1 =', accuracy1)
    print('accuracy2 =', accuracy2)

    print('end_time =', datetime.datetime.now())
