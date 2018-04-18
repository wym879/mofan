import tensorflow as tf

# 添加神经层
def add_layer(inputs, in_size, out_size, activation_fucntion = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_fucntion is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_fucntion(Wx_plus_b)
    return outputs

