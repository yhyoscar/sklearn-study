import tensorflow as tf

def deepnn(x):
    x = tf.reshape(x, [-1, 28, 28, 1])
    w1 = tf.Variable(tf.truncated_normal([3,3,1,4], stddev=0.1), name='w1')
    b1 = tf.Variable(tf.constant(0.1, shape=[4]), name='b1')
    c1 = tf.nn.relu(tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
    p1 = tf.nn.max_pool(c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w2 = tf.Variable(tf.truncated_normal([14*14*4, 25], stddev=0.1), name='w2')
    b2 = tf.Variable(tf.constant(0.1, shape=[25]), name='b2')
    x2 = tf.nn.relu(tf.matmul(tf.reshape(p1, [-1, 14*14*4]), w2) + b2)

    keep_prob = tf.placeholder(tf.float32)
    x2_drop = tf.nn.dropout(x2, keep_prob)
    
    w3 = tf.Variable(tf.truncated_normal([25,10], stddev=0.1), name='w3')
    b3 = tf.Variable(tf.constant(0.1, shape=[10]), name='b3')
    ye = tf.matmul(x2_drop, w3) + b3
    return ye, keep_prob

x = tf.placeholder(tf.float32, [None, 28*28])
y = tf.placeholder(tf.float32, [None, 10])
ye, keep_prob = deepnn(x)
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=ye))
model = tf.train.AdamOptimizer(1e-2).minimize(error)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ye, axis=1), tf.argmax(y, axis=1)), tf.float32))
#tf.summary.FileWriter('./', graph=tf.get_default_graph())

#=====================================================
from time import time as timer
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data', one_hot=True)
nbatch = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t0 = timer()
    for i in range(1000):
        batch = mnist.train.next_batch(nbatch)
        #ac = accuracy.eval(feed_dict={x:batch[0], y:batch[1], keep_prob:1.0})
        ac = accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        t1 = timer()
        print 'step: ', i, ', accuracy: ', ac, ', time: ', t1-t0
        t0 = timer()
        
        model.run(feed_dict={x:batch[0], y:batch[1], keep_prob:1.0})

