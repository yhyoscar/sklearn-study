import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 12], name='X')
y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

w = tf.Variable(tf.truncated_normal([12, 1], stddev=0.1), name='W')
b = tf.Variable(tf.constant(0.1, shape=[1]), name='b')

yhat = tf.nn.relu(tf.matmul(x, w) + b)

error = tf.reduce_sum(tf.square(y - yhat))

tf.summary.FileWriter('./', graph=tf.get_default_graph())

