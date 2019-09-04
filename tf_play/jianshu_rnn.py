import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 获取MNIST数据
mnist = input_data.read_data_sets(r"./MNIST_data", one_hot=True)

# 设置RNN结构
element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 150

# 输入向量和输出向量
# 定义张数，循环数，单个元素长度
_inputs = tf.placeholder(tf.float32, shape=[None, time_steps, element_size], name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='inputs')

# 利用TensorFlow的内置函数BasicRNNCell, dynamic_rnn来构建RNN的基本模块
# 定义每个隐藏层的节点是150个，这意味着W是150*28
rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)
# 这里有一个很NB的地方，直接告知输出节点的数量就可以了，参数不输入不输出，自动训练
# 本例返回100*28*150的矩阵
outputs, _ = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)

# 这里只定义最后一层的特征向量
Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], mean=0,stddev=.01))
bl = tf.Variable(tf.truncated_normal([num_classes],mean=0,stddev=.01))

# 取输出的向量outputs中的最后一个向量最为最终输出
last_rnn_output = outputs[:,-1,:]
final_output = tf.matmul(last_rnn_output, Wl) + bl

# 定义损失函数并用RMSPropOptimizer优化
# 接下来就是简单的softmax和梯度下降法，和FC一样。这样一个RNN就完成了
softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=y)
cross_entropy = tf.reduce_mean(softmax)
# 我觉得有空还是要学一下RMS和ADAM，先办主要的事
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

# 统计准确率
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(final_output,1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# 测试集
test_data = mnist.test.images[:batch_size].reshape((-1, time_steps, element_size))
test_label = mnist.test.labels[:batch_size]

# 每次训练batch_size张图片，一共训练3000次
for i in range(3001):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size, time_steps, element_size))
    sess.run(train_step, feed_dict={_inputs:batch_x, y:batch_y})
    if i % 100 == 0:
        loss = sess.run(cross_entropy, feed_dict={_inputs: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={_inputs:batch_x, y: batch_y})
        print ("Iter " + str(i) + ", Minibatch Loss= " + \
               "{:.6f}".format(loss) + ", Training Accuracy= " + \
               "{:.5f}".format(acc))

# 在测试集上的准确率
print("Testing Accuracy:", sess.run(accuracy, feed_dict={_inputs:test_data, y:test_label}))
