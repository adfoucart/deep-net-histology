import os
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy import random
import csv

import tensorflow as tf
import json
from datetime import datetime
#from MITOS12Feed import MITOS12Feed

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

def weights_xavier(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


# def add_residual_unit(net, layers):
#     net_long = net
#     for l in layers:
#         net_long = tf.contrib.layers.conv2d(net_long, l['channels'], l['ksize'], 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope=l['name'])
#     return tf.add(net, net_long)

'''
RESNET
    # Widen the network
    net1 = tf.contrib.layers.conv2d(X, 256, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    # Residual unit 1
    net2a = tf.contrib.layers.conv2d(net1, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_1')
    net2b = tf.contrib.layers.conv2d(net2a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_2')
    net2c = tf.contrib.layers.conv2d(net2b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_3')
    net2 = tf.add(net1, net2c)

    # Residual unit 2
    net3a = tf.contrib.layers.conv2d(net2, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_1')
    net3b = tf.contrib.layers.conv2d(net3a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_2')
    net3c = tf.contrib.layers.conv2d(net3b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_3')
    net3 = tf.add(net2, net3c)

    # Residual unit 3
    net4a = tf.contrib.layers.conv2d(net3, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_1')
    net4b = tf.contrib.layers.conv2d(net4a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_2')
    net4c = tf.contrib.layers.conv2d(net4b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_3')
    net4 = tf.add(net3, net4c)

    # Residual unit 4
    net5a = tf.contrib.layers.conv2d(net4, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_1')
    net5b = tf.contrib.layers.conv2d(net5a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_2')
    net5c = tf.contrib.layers.conv2d(net5b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_3')
    net5 = tf.add(net4, net5c)

    net6 = tf.contrib.layers.conv2d(net5, 64, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow1')
    net = tf.contrib.layers.conv2d(net6, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow')
'''

'''
RESNET_SHORT
    # Widen the network
    net1 = tf.contrib.layers.conv2d(X, 256, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    # Residual unit 1
    net2a = tf.contrib.layers.conv2d(net1, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_1')
    net2b = tf.contrib.layers.conv2d(net2a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_2')
    net2c = tf.contrib.layers.conv2d(net2b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_3')
    net2 = tf.add(net1, net2c)

    # Residual unit 2
    net3a = tf.contrib.layers.conv2d(net2, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_1')
    net3b = tf.contrib.layers.conv2d(net3a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_2')
    net3c = tf.contrib.layers.conv2d(net3b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_3')
    net3 = tf.add(net2, net3c)

    net6 = tf.contrib.layers.conv2d(net3, 64, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow1')
    net = tf.contrib.layers.conv2d(net6, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow')
'''

'''
RESNET_DEEP
    # Widen the network
    net1 = tf.contrib.layers.conv2d(X, 256, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    # Residual unit 1
    net2a = tf.contrib.layers.conv2d(net1, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/1')
    net2b = tf.contrib.layers.conv2d(net2a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/2')
    net2c = tf.contrib.layers.conv2d(net2b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/3')
    net2 = tf.add(net1, net2c, name='res1/add')

    # Residual unit 2
    net3a = tf.contrib.layers.conv2d(net2, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/1')
    net3b = tf.contrib.layers.conv2d(net3a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/2')
    net3c = tf.contrib.layers.conv2d(net3b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/3')
    net3 = tf.add(net2, net3c, name='res2/add')

    # Residual unit 3
    net4a = tf.contrib.layers.conv2d(net3, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/1')
    net4b = tf.contrib.layers.conv2d(net4a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/2')
    net4c = tf.contrib.layers.conv2d(net4b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/3')
    net4 = tf.add(net3, net4c, name='res3/add')

    # Residual unit 4
    net5a = tf.contrib.layers.conv2d(net4, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/1')
    net5b = tf.contrib.layers.conv2d(net5a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/2')
    net5c = tf.contrib.layers.conv2d(net5b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/3')
    net5 = tf.add(net4, net5c, name='res4/add')

    # Residual unit 5
    net6a = tf.contrib.layers.conv2d(net5, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res5/1')
    net6b = tf.contrib.layers.conv2d(net6a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res5/2')
    net6c = tf.contrib.layers.conv2d(net6b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res5/3')
    net6 = tf.add(net5, net6c, name='res5/add')

    # Residual unit 6
    net7a = tf.contrib.layers.conv2d(net6, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res6/1')
    net7b = tf.contrib.layers.conv2d(net7a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res6/2')
    net7c = tf.contrib.layers.conv2d(net7b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res6/3')
    net7 = tf.add(net6, net7c, name='res6/add')

    # Residual unit 7
    net8a = tf.contrib.layers.conv2d(net7, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res7/1')
    net8b = tf.contrib.layers.conv2d(net8a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res7/2')
    net8c = tf.contrib.layers.conv2d(net8b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res7/3')
    net8 = tf.add(net7, net8c, name='res7/add')

    # Residual unit 8
    net9a = tf.contrib.layers.conv2d(net8, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res8/1')
    net9b = tf.contrib.layers.conv2d(net9a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res8/2')
    net9c = tf.contrib.layers.conv2d(net9b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res8/3')
    net9 = tf.add(net8, net9c, name='res8/add')

    net10 = tf.contrib.layers.conv2d(net9, 64, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow1')
    net = tf.contrib.layers.conv2d(net10, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow')
'''

def get_network(X):
    batch_size = X.get_shape()[0]
    im_size = tf.constant([X.get_shape().as_list()[1], X.get_shape().as_list()[2]])
    channels = X.get_shape()[3]
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0

    # Widen the network
    net1 = tf.contrib.layers.conv2d(X, 256, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    # Residual unit 1
    net2a = tf.contrib.layers.conv2d(net1, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_1')
    net2b = tf.contrib.layers.conv2d(net2a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_2')
    net2c = tf.contrib.layers.conv2d(net2b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_3')
    net2 = tf.add(net1, net2c)

    # Residual unit 2
    net3a = tf.contrib.layers.conv2d(net2, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_1')
    net3b = tf.contrib.layers.conv2d(net3a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_2')
    net3c = tf.contrib.layers.conv2d(net3b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_3')
    net3 = tf.add(net2, net3c)

    # Residual unit 3
    net4a = tf.contrib.layers.conv2d(net3, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_1')
    net4b = tf.contrib.layers.conv2d(net4a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_2')
    net4c = tf.contrib.layers.conv2d(net4b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_3')
    net4 = tf.add(net3, net4c)

    # Residual unit 4
    net5a = tf.contrib.layers.conv2d(net4, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_1')
    net5b = tf.contrib.layers.conv2d(net5a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_2')
    net5c = tf.contrib.layers.conv2d(net5b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_3')
    net5 = tf.add(net4, net5c)

    net6 = tf.contrib.layers.conv2d(net5, 64, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow1')
    net = tf.contrib.layers.conv2d(net6, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow')

    return [net]
    #return net,net6,net5,net5c,net4,net4c,net3,net3c,net2,net2c,net1

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def feed(path, seed=0):
    np.random.seed(seed)

    Xfiles = [f for f in os.listdir(path) if f.find('patches') >= 0]
    Yfiles = [f for f in os.listdir(path) if f.find('targets') >= 0]

    N = len(Xfiles)
    idxs = np.arange(N)
    np.random.shuffle(idxs)
    
    for idx in idxs:
        yield np.load(os.path.join(path, Xfiles[idx])), np.load(os.path.join(path, Yfiles[idx]))[:,:,:,np.newaxis]


imscale = lambda X : (X-X.min())/(X.max()-X.min()) 

def test(saver, batches, net, sess, clf_name, loss, batch_size=20):
    saver.restore(sess, "e:/data/tf_checkpoint/%s.ckpt"%clf_name)
    batch_size = min(batch_size, 20)

    epoch = 1
    for batch in feed(batches, epoch):
        t = batch[1].astype('float')
        # t[t==0] = -1.
        t[t>0] = 1.
        Y = net.eval(session=sess, feed_dict={X: batch[0][:batch_size], target: t[:batch_size]})
        # loss = sess.run(loss, feed_dict={X: batch[0], target: t})
        #print(loss, np.sqrt((Y-t)**2).sum()/(20*127*127))
        plt.figure()
        for i in range(min(batch_size,10)):
            im = imscale(batch[0][i])
            plt.subplot(2,5,i+1)
            plt.imshow(im)
        plt.figure()
        for i in range(min(batch_size,10)):
            plt.subplot(2,5,i+1)
            plt.imshow(batch[1][i][:,:,0])
        plt.figure()
        for i in range(min(batch_size,10)):
            plt.subplot(2,5,i+1)
            plt.imshow(Y[i][:,:,0])
        plt.figure()
        for i in range(min(batch_size,10)):
            plt.subplot(2,5,i+1)
            plt.imshow(Y[i][:,:,0]>0.5)
        plt.show()
        break

def train(saver, batches, net, sess, clf_name, loss, merged, train_writer, restore_from=None, batch_size=20):
    c = 10000
    i = 0
    batch_size = min(batch_size, 20)

    print("Start training")

    if restore_from != None:
        saver.restore(sess, "e:/data/tf_checkpoint/%s.ckpt"%restore_from)

    for epoch in range(4):
        print("Epoch %d"%(epoch+1))
        for batch in feed(batches, epoch):
            t = batch[1].astype('float')
            t[t>0] = 1.
            trainingStep.run(session=sess, feed_dict={X: batch[0][:batch_size], target: t[:batch_size]})
            if( i % 100 == 0 ):
                [summ, lv] = sess.run([merged,loss], feed_dict={X: batch[0][:batch_size], target: t[:batch_size]})
                train_writer.add_summary(summ, i)
                saver.save(sess, "e:/data/tf_checkpoint/%s.ckpt"%clf_name)
            i += 1

        saver.save(sess, "e:/data/tf_checkpoint/%s.ckpt"%clf_name)

import csv
def get_ground_truth(super_file):
    mask = np.zeros((2084,2084))
    with open(super_file, 'r', encoding="utf8") as f:
        reader = csv.reader(f, delimiter=',')
        mitosis_raw = [np.array(row).astype('int') for row in reader]
        mitosis_coord = [np.vstack([m[::2], m[1::2]]) for m in mitosis_raw]
        for c in mitosis_coord:
            mask[c[1],c[0]] = 1
        return mask
    return None

def get_prob_mask():
    # Mask for adding probs: 1 at the center, decreasing to 0 at the corners
    P = np.zeros((127,127))
    for x in range(127):
        for y in range(127):
            P[y,x] = (y-63)**2+(x-63)**2
    P = (P.max()-P)/(P.max()-P.min())
    return P

def run_complete_test(saver, net, sess, clf_name, just_one=False):
    testdir = 'e:/data/MITOS12/test'
    patchesdir = os.path.join(testdir, 'patches')
    clf = "e:/data/tf_checkpoint/%s.ckpt"%clf_name

    images = [f for f in os.listdir(testdir) if f.find('.bmp') >= 0]

    P = get_prob_mask()
    # thresh = 0.91 # Threshold determined from the training set
    saver.restore(sess, clf)

    for imfile in images:
        print(imfile)
        patches = [f for f in os.listdir(patchesdir) if f.find(imfile) >= 0]
        classes = np.zeros((2084,2084))
        ns = np.zeros((2084,2084))
        csv = os.path.join(testdir, imfile.replace('bmp','csv'))
        GT = get_ground_truth(csv)
        for p in patches:
            y = int(p[11:15])*10
            patch = np.load(os.path.join(patchesdir,p)).astype('float')/255
            
            Y = np.zeros((196,127,127,1))
            for k in range(7):
                Y[k*28:(k+1)*28] = net.eval(session=sess, feed_dict={X: patch[k*28:(k+1)*28]})

            for x in range(196):
                classes[y:y+127,x*10:(x*10)+127] += Y[x,:,:,0]*P
                ns[y:y+127,x*10:(x*10)+127] += P

            print(int(p[11:15]),'/',195)

        classes[ns>0.5] /= ns[ns>0.5]
        classes[ns<= 0.5] = 0
        np.save('%s.%s.npy'%(clf,imfile), classes)

        if just_one: break
        #M = classes[:,:,1]>=thresh
        #true_centroids = [r.centroid for r in regionprops(label(GT))]
        #clf_centroids = [r.centroid for r in regionprops(label(M))]


def detailed_test(nets, clf_name, batches):
    saver.restore(sess, "e:/data/tf_checkpoint/%s.ckpt"%clf_name)

    net,net6,net5,net5c,net4,net4c,net3,net3c,net2,net2c,net1 = nets

    for batch in feed(batches, 0):
        x = batch[0]
        y = batch[1].astype('float')
        y[y>0] = 1.
        np.save('%s_Y.npy'%clf_name, y)
        np.save('%s_X.npy'%clf_name, x)
        np.save('%s_net1.npy'%clf_name, net1.eval(session=sess, feed_dict={X: x[:batch_size], target: y[:batch_size]}))
        np.save('%s_net2.npy'%clf_name, net2.eval(session=sess, feed_dict={X: x[:batch_size], target: y[:batch_size]}))
        np.save('%s_net3.npy'%clf_name, net3.eval(session=sess, feed_dict={X: x[:batch_size], target: y[:batch_size]}))
        np.save('%s_net4.npy'%clf_name, net4.eval(session=sess, feed_dict={X: x[:batch_size], target: y[:batch_size]}))
        np.save('%s_net5.npy'%clf_name, net5.eval(session=sess, feed_dict={X: x[:batch_size], target: y[:batch_size]}))
        np.save('%s_net6.npy'%clf_name, net6.eval(session=sess, feed_dict={X: x[:batch_size], target: y[:batch_size]}))
        np.save('%s_net.npy'%clf_name, net.eval(session=sess, feed_dict={X: x[:batch_size], target: y[:batch_size]}))
        break

import sys
import datetime
if __name__ == "__main__":
    # Get command line argument : train / test / detailed / fulltest

    if( len(sys.argv) > 1 ):
        action = sys.argv[1]
        batch_size = 28 if action == 'fulltest' else 20            # Use 28 for final test, 20 for training
        im_size = (127,127)
        lr = 1e-2
        eps = 0.1
        a = 1e-4

        clf_name = "resnet_short" if len(sys.argv) <= 2 else sys.argv[2]
        clf_from = None if len(sys.argv) <= 3 else sys.argv[3] 

        batches = "E:\\data\\MITOS12\\train\\batches_with_stuff"
        # batches = "E:\\data\\MITOS12\\train\\batches"

        X = tf.placeholder(tf.float32, [batch_size,im_size[0],im_size[1],3])

        #net,net6,net5,net5c,net4,net4c,net3,net3c,net2,net2c,net1 = get_network(X)
        nets = get_network(X)
        net = nets[0]

        saver = tf.train.Saver()

        target = tf.placeholder(tf.float32, [batch_size, im_size[0],im_size[1],1])
        loss = tf.losses.mean_squared_error(target, net)
        loss = tf.add_n([loss] + [a*r for r in tf.losses.get_regularization_losses()])
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()

        optimizer = tf.train.AdamOptimizer(lr, epsilon=eps)
        trainingStep = optimizer.minimize(loss)

        sess = tf.Session()
        train_writer = tf.summary.FileWriter('./summaries/train/%s'%(clf_name), sess.graph)
        sess.run(tf.global_variables_initializer())

        tf.get_default_graph().finalize()
        
        if action == 'build':
            print("Built network.")
        elif action == 'train':
            train(saver, batches, net, sess, clf_name, loss, merged, train_writer, clf_from, batch_size)
        elif action == 'test':
            test(saver, batches, net, sess, clf_name, loss, batch_size)
        elif action == 'detailed':
            detailed_test(nets, clf_name, batches)
        elif action == 'fulltest':
            run_complete_test(saver, net, sess, clf_name, True)
        else:
            print("Unknow command.")
            print("Usage : python resnet.py train|test|detailed|fulltest [clf_name] [clf_from]")

    else:
        print("Usage : python resnet.py train|test|detailed|fulltest [clf_name] [clf_from]")
