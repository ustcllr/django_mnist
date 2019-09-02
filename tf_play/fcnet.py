"""
全连接网络的实验
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

# 定义节点数列表
# 注：TF的shape也是一个张量，要转成int
n = [int(x_train.shape[1]), 100, 50, 10]

# 定义层的总数
L = len(n) - 1

# 定义循环次数
times = 10

# 定义机器学习率
lr = 1

# 定义mini-batch大小
mini_size = 1000

# 得到batch数目
batch_num = int(60000 / mini_size)


x_tf = tf.placeholder(tf.float32, [None, 784])
y_tf = tf.placeholder(tf.float32, [None, 10])

# 初始化向量
w = [tf.zeros([1])] * (L+1)
b = [tf.zeros([1])] * (L+1)
z = [tf.zeros([1])] * (L+1)
a = [tf.zeros([1])] * (L+1)
a[0] = x_tf

# 对于RELU，尽量控制输出层的方差也为1，对于逻辑回归，都设在原点附近
for l in range(1, L+1):
    # 对于RELU，第l层的初始化常数与第l-1层的节点数相关
    if l < L:
        ir = tf.sqrt(2 / n[l-1])
    else:
        ir = 0.01
    w[l] = tf.Variable(tf.random.normal([n[l-1], n[l]]) * ir)
    b[l] = tf.Variable(tf.zeros([1, n[l]]))

# 前向传播：前面L-1层使用RELU，第L层使用Softmax
for l in range(1, L+1):
    z[l] = tf.matmul(a[l-1], w[l]) + b[l]
    if l < L:
        a[l] = tf.nn.relu(z[l])
    else:
        a[l] = tf.nn.softmax(z[l])

# 预防出现log0的问题，这在训练后期经常发生
a_clip = tf.clip_by_value(a[L], 1e-10, 1.0)
cost = tf.reduce_sum(- y_tf * tf.math.log(a_clip)) / mini_size
train = tf.compat.v1.train.GradientDescentOptimizer(lr).minimize(cost)

# 根据xy，计算模型的准确率
correct_prediction = tf.equal(tf.argmax(a[L], 1), tf.argmax(y_tf, 1))
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
            if (t+1) % 1 == 0 and i == 10:
                cost_ran = session.run(cost, feed_dict={x_tf: x_mini, y_tf: y_mini})
                accuracy_ran = session.run(accuracy, feed_dict={x_tf: x_mini, y_tf: y_mini})
                print('t = {}, j = {}, accuracy = {}'.format(t+1, cost_ran, accuracy_ran))

    # 看一看模型的准确率
    accuracy1 = session.run(accuracy, feed_dict={x_tf: x_norm_train, y_tf: y_train})
    accuracy2 = session.run(accuracy, feed_dict={x_tf: x_norm_test, y_tf: y_test})
    print('accuracy1 =', accuracy1)
    print('accuracy2 =', accuracy2)

    print('end_time =', datetime.datetime.now())
