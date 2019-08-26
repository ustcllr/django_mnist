"""
使用卷积神经网络，将识别率提升到99%
"""

import os
import datetime
import json

import numpy as np
import tensorflow as tf

from data_support import get_training_image_ary, get_training_label_ary, \
        get_recognize_image_ary, get_recognize_label_ary, \
        TRAINING_SUM, RECOGNIZE_SUM, DATASET_NAME
from utils import forward_prop, back_prop, get_normalize, get_bias


# 得到训练集和测试集
x_train = get_training_image_ary()
y_train = get_training_label_ary()
x_test = get_recognize_image_ary()
y_test = get_recognize_label_ary()

# 对训练集输入向量归一化
mu_x, sigma_x, x_norm_train = get_normalize(x_train)
x_norm_train = x_norm_train.astype(np.float32)

# 对测试集输入向量归一化
x_norm_test = (x_test - mu_x) / sigma_x
x_norm_test = x_norm_test.astype(np.float32)

# 定义训练集的总数
m = TRAINING_SUM

# 定义节点数列表
n = [x_train.shape[0], 100, 50, 10]

# 定义层的总数
L = len(n) - 1

# 定义循环次数
times = 10

# 定义机器学习率
lr = 0.0001

# 定义mini-batch大小
mini_size = 100

# 得到batch数目
batch_num = int(m / mini_size)


x_tf = tf.placeholder('float', [None, 784])
y_tf = tf.placeholder('float', [None, 10])

# 将输入向量从2维变成4维
x_input = tf.reshape(x_tf, [-1, 28, 28, 1])

# 根据TF的规则，过滤器按照这个维度顺序初始化
w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))

# 根据规则，b只设定一个维度
b1 = tf.Variable(tf.constant(0.1, shape=[32, ]))

# 得到第1个卷积层
z1 = tf.nn.conv2d(x_input, w1, 1, 'SAME') + b1
a1 = tf.nn.relu(z1)

# 得到第1个最大池化层
max_pool1 = tf.nn.max_pool2d(a1, [2, 2], 2, 'SAME')

# 初始化第2个卷积层权重
w2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[64, ]))

# 得到第2个卷积层
z2 = tf.nn.conv2d(max_pool1, w2, 1, 'SAME') + b2
a2 = tf.nn.relu(z2)
max_pool2 = tf.nn.max_pool2d(a2, [2, 2], 2, 'SAME')

# 展开成全连接节点
a_fc0 = tf.reshape(max_pool2, [-1, 7*7*64])

# 第1个FC层
w_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.zeros([1024]))
z_fc1 = tf.matmul(a_fc0, w_fc1) + b_fc1
a_fc1 = tf.nn.relu(z_fc1)

# 第2个FC层，进行分类
w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.zeros([10]))
z_fc2 = tf.matmul(a_fc1, w_fc2) + b_fc2
a_fc2 = tf.nn.softmax(z_fc2)

# cost = - tf.reduce_mean(y_tf * tf.math.log(a_fc2))
cost = - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_tf, logits=a_fc2))

# # 学习率衰减
# global_step = tf.placeholder(tf.int32)
# lr_opti = tf.train.exponential_decay(
#     learning_rate=lr,
#     global_step=global_step,
#     decay_steps=5,
#     decay_rate=0.9,
#     staircase=True
# )
train = tf.train.AdamOptimizer(lr).minimize(cost)

# 根据xy，计算模型的准确率
correct_prediction = tf.equal(tf.argmax(a_fc2, 1), tf.argmax(y_tf, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# 开始训练模型
init = tf.initialize_all_variables()
with tf.compat.v1.Session() as session:
    # 激活TF变量
    session.run(init)
    # 打印分隔线
    print('\n{}'.format('-'*70))
    print('begin_time =', datetime.datetime.now())

    for t in range(times):
        for i in range(batch_num):
            x_mini = x_norm_train.T[i*mini_size: (i+1)*mini_size]
            y_mini = y_train.T[i*mini_size: (i+1)*mini_size]

            session.run(train, feed_dict={x_tf: x_mini, y_tf: y_mini})

            # 每训练10次，计算成本函数，观察梯度是否下降
            if (t+1) % 1 == 0 and i == 10:
                print('t = {}, j = {}'.format(t+1, session.run(cost,
                        feed_dict={x_tf: x_mini, y_tf: y_mini})))

    # 看一看模型的准确率
    accuracy1 = session.run(accuracy, feed_dict={x_tf: x_norm_train.T, y_tf: y_train.T})
    accuracy2 = session.run(accuracy, feed_dict={x_tf: x_norm_test.T, y_tf: y_test.T})
    print('accuracy1 =', accuracy1)
    print('accuracy2 =', accuracy2)

    print('end_time =', datetime.datetime.now())
