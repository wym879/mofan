import tensorflow as tf
import numpy as np

# 添加神经层
def add_layer(inputs, in_size, out_size, activation_fucntion = None):
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_fucntion is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_fucntion(Wx_plus_b)
        return outputs


# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis] # 添加维度
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_fucntion=tf.nn.relu) # 10个神经元
# add output layer
predition = add_layer(l1, 10, 1, activation_fucntion=None) # 线性输出


# the error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition),
                                        reduction_indices=[1])) # TODO: reduction_indices ???
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
writer = tf.summary.FileWriter('logs/', sess.graph)     # 可视化文件,放到
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data}) # 假设用全部的数据
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))