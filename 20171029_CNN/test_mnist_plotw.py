import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('../data', one_hot=True)

nx = 28; ny = 28; nchannel = 1; nlabel = 10

cnum    = [4, 8]     # convolutional layer filters
csize   = [5, 7]     # filter size
cstride = [1, 1]     # filter stride
psize   = [3, 3]     # pooling size
pstride = [3, 3]     # pooling stride

fnum  = [30]       # full connected layer
pkeep = 1.0     # drop out: keep probability

#------------------------------------------------------------------------
nconv = len(cnum)
nfull = len(fnum)

x = tf.placeholder(tf.float32, [None, nx*ny*nchannel])
y = tf.placeholder(tf.float32, [None, nlabel])
keep_prob = tf.placeholder(tf.float32) 

#------------------------------------------------------------------------
layer = [ tf.reshape(x, [-1, nx, ny, nchannel]) ]
w = []; b = []
k = 0
for i in range(nconv):
    if i==0: nfilter = nchannel
    else:    nfilter = cnum[i-1]
    w.append( tf.Variable(tf.truncated_normal([csize[i], csize[i], nfilter, cnum[i]], stddev=0.1)) )
    b.append( tf.Variable(tf.constant(0.1, shape=[cnum[i]])) )
    print 'w,b: ', w[-1].shape, b[-1].shape
    layer.append( tf.nn.relu(tf.nn.conv2d(layer[k], w[i], strides=[1, cstride[i], cstride[i], 1], padding='SAME') + b[i]) )
    k += 1
    print 'layer: ', layer[-1]
    layer.append( tf.nn.max_pool(layer[k], ksize=[1, psize[i], psize[i], 1], strides=[1, pstride[i], pstride[i], 1], padding='VALID') )
    k += 1
    print 'layer: ', layer[-1]

nlast = int(layer[k].shape[-1]*layer[k].shape[-2]*layer[k].shape[-3])
layer.append( tf.reshape(layer[k], [-1, nlast]) )
k += 1
print 'layer: ', layer[-1]

for i in range(nfull):
    if i==0: nnode = nlast
    else:    nnode = fnum[i-1]
    w.append( tf.Variable(tf.truncated_normal([nnode, fnum[i]], stddev=0.1)) )
    b.append( tf.Variable(tf.constant(0.1, shape=[fnum[i]])) )
    print 'w,b: ', w[-1].shape, b[-1].shape
    layer.append(tf.nn.dropout(tf.nn.relu(tf.matmul(layer[k], w[-1]) + b[-1]) , keep_prob ) )
    k += 1
    print 'layer: ', layer[-1]

w.append( tf.Variable(tf.truncated_normal([fnum[-1], nlabel], stddev=0.1)) )
b.append( tf.Variable(tf.constant(0.1, shape=[nlabel])) )
print 'w,b: ', w[-1].shape, b[-1].shape
layer.append( tf.matmul(layer[k], w[-1]) + b[-1] )
print 'layer: ', layer[-1]

error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=layer[-1]))
model = tf.train.AdamOptimizer(1e-2).minimize(error)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(layer[-1], axis=1), tf.argmax(y, axis=1)), tf.float32))
#tf.summary.FileWriter('./', graph=tf.get_default_graph())



def fplotw(data, ifig=1, ffig=''):
    ndim = len(data.shape)
    if ndim == 2: 
        plt.figure(ifig, figsize=(data.shape[1]/4.0,data.shape[0]/4.0))
        nrow = 1; ncol = 1
        data = data.reshape([data.shape[0], data.shape[1], 1, 1])
    if ndim == 4: 
        plt.figure(ifig, figsize=(data.shape[2]*2,data.shape[3]*2))
        nrow = data.shape[3]; ncol = data.shape[2]
    for i in range(nrow):
        for j in range(ncol):
            plt.subplot(nrow,ncol,i*ncol+j+1)
            plt.imshow(data[:,:,j,i], interpolation = 'None', cmap='gray')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig(ffig)
    plt.close(ifig)
    return


from time import time as timer
import numpy as np
matplotlib.use('Agg')

nbatch = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t0 = timer()
    for i in range(100):
        batch = mnist.train.next_batch(nbatch)
        #ac = accuracy.eval(feed_dict={x:batch[0], y:batch[1], keep_prob:1.0})
        ac = accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        t1 = timer()
        print 'step: ', i, ', accuracy: ', ac, ', time: ', t1-t0
        t0 = timer()        
        model.run(feed_dict={x:batch[0], y:batch[1], keep_prob:pkeep})
        
        for iw in range(len(w)):
            data = sess.run(w[iw])
            fplotw(data, ifig=iw+1, ffig='../figures/test_t'+format(i+1,'06')+'_w'+format(iw+1)+'.png')
        

