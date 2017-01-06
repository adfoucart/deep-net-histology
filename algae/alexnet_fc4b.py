import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import csv

from EqualizedDataFeed import EqualizedDataFeed
from DataFeed import DataFeed

import tensorflow as tf
import json
from datetime import datetime

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


class AlexNet:
    def __init__(self, batch_size, im_size, sess, lr, eps, a):
        net_data = np.load("../alexnet/bvlc_alexnet.py35.npy").item()

        self.x = tf.placeholder(tf.float32, [batch_size,im_size[0],im_size[1],3])
        res = tf.image.resize_images(self.x, (227, 227))

        #conv1 / kernel : (11, 11, 96) / stride : (4, 4) / name='conv1'
        conv1W = tf.Variable(net_data["conv1"][0])
        conv1b = tf.Variable(net_data["conv1"][1])
        conv1 = tf.nn.relu(conv(res, conv1W, conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1))

        #local response normalization 1
        #lrn(2, 2e-05, 0.75, name='norm1')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)

        #maxpool1
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')


        #conv2 / (5, 5, 256) / (1, 1)
        conv2W = tf.Variable(net_data["conv2"][0])
        conv2b = tf.Variable(net_data["conv2"][1])
        conv2 = tf.nn.relu(conv(maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding="SAME", group=2))


        #lrn2
        #lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)

        #maxpool2                                                
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        #conv3 / (3, 3, 384) / (1, 1)
        conv3W = tf.Variable(net_data["conv3"][0])
        conv3b = tf.Variable(net_data["conv3"][1])
        conv3 = tf.nn.relu(conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1))

        #conv4 / (3, 3, 384) / (1, 1)
        conv4W = tf.Variable(net_data["conv4"][0])
        conv4b = tf.Variable(net_data["conv4"][1])
        conv4 = tf.nn.relu(conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2))

        # New layers : FullyConnected - Softmax
        conv4r = tf.reshape(conv4, [batch_size, int(np.prod(conv4.get_shape()[1:]))])

        fc5W = tf.Variable(tf.truncated_normal([int(np.prod(conv4.get_shape()[1:])),4], stddev=0.1))
        fc5b = tf.Variable(tf.zeros([4]))
        fc5 = tf.nn.relu6(tf.matmul(conv4r, fc5W) + fc5b)

        # Softmax
        softmax = tf.nn.softmax(fc5)

        self.output = softmax

        # Setup Training
        self.train_W = [fc5W]
        self.train_b = [fc5b]
        self.train_vars = self.train_W+self.train_b
        self.target = tf.placeholder(tf.float32, [batch_size, 4])

        # Using cross-entropy
        self.loss = -tf.reduce_mean(self.target*tf.log(self.output) + (1-self.target)*tf.log(1-self.output))
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.train_W])
        self.cost = self.loss+a*self.l2_loss
        optimizer = tf.train.AdamOptimizer(lr, epsilon=eps)
        self.trainingStep = optimizer.minimize(self.cost, var_list=self.train_vars)

        self.sess = sess
        self.sess.run(tf.initialize_all_variables())

    def forward(self, X):
        return self.sess.run(self.output, feed_dict={self.x:X})

    def train(self, X, Y):
        self.trainingStep.run(feed_dict={self.x: X, self.target: Y})
    
    def get_cost(self, X, Y):
        return (self.loss.eval(feed_dict={self.x: X, self.target: Y}), self.l2_loss.eval(feed_dict={self.x: X, self.target: Y}))

def train():
    clf_name = "AlexNet_4_1fc_1softmax"
    feed = EqualizedDataFeed("e:/data/algae_dataset_equal_batches", False)
    sess = tf.InteractiveSession()
    batch_size = 200
    tile_size = (128,128)
    lr = 1e-1
    eps = 0.1
    a = 0.001
    net = AlexNet(batch_size, tile_size, sess, lr, eps, a)
    saver = tf.train.Saver(net.train_vars)
    best_score = 5.

    now = datetime.now().strftime("%Y%m%d%H%M")
    clf_date_name = '%s_%s'%(clf_name, now)
    with open('./%s.log'%clf_date_name, 'w') as logfile:
        run_info = {'clf_name': clf_name,
                    'batch_size': batch_size,
                    'tile_size': tile_size,
                    'description': "AlexNet, cut at level 4, 1 fully connected relu6 and 1 softmax.",
                    'file': "alexnet_fc4.py",
                    'lr': lr,
                    'eps': eps,
                    'a': a,
                    'continuing_from_previous_run': False,
                    # 'restored_clf': "AlexNet_4ft_1mp_2fc_1softmax_201610131540_last.ckpt",
                    'Date': now}
        json.dump(run_info, logfile, indent=4)
        logfile.write("\n")

    # saver.restore(sess, "AlexNet_4ft_1mp_2fc_1softmax_201610131540_last.ckpt")
    # if(os.path.isdir('weights/%s'%clf_date_name) == False):
    #     os.mkdir('weights/%s'%clf_date_name)

    print("Start learning %s..."%clf_date_name)
    for i in range(20001):
        if i%10==0:
            X,Y = feed.next_batch(batch_size)
            cost = net.get_cost(X,Y)
            with open('%s_%s.log'%(clf_name, now), 'a') as logfile:
                logfile.write("Cost at iteration %d : %f + a*%f\n"%(i, cost[0], cost[1]))
            if cost[0] < best_score:
                print("Best score : %f."%cost[0])
                best_score = cost[0]
                saver.save(sess, "./%s_best.ckpt"%clf_date_name)
                # np.save('weights/%s/wtop_best.npy'%clf_date_name, net.train_W[0].eval())
                # np.save('weights/%s/wbottom_best.npy'%clf_date_name, net.train_W[-1].eval())
            # np.save('weights/%s/wtop_%05d.npy'%(clf_date_name,i), net.train_W[0].eval())

        X,Y = feed.next_batch(batch_size)
        net.train(X,Y)

    saver.save(sess, "./%s_last.ckpt"%clf_date_name)

def test():
    clf_name = "AlexNet_4_1fc_1softmax"
    # feed = DataFeed("/mnt/e/data/algae_dataset", "/mnt/e/data/algae_patches", False)
    feed = DataFeed("/mnt/e/data/algae_dataset_cells_only", "/mnt/e/data/algae_patches_cells_only", False, False)
    # feed = EqualizedDataFeed("/mnt/e/data/algae_dataset_equal_batches", False)
    sess = tf.InteractiveSession()
    batch_size = 200
    tile_size = (128,128)
    lr = 1e-3
    eps = 1.0
    a = 0.001
    net = AlexNet(batch_size, tile_size, sess, lr, eps, a)
    saver = tf.train.Saver(net.train_vars)

    saver.restore(sess, "AlexNet_4_1fc_1softmax_201610311614_best.ckpt")

    conf_mat = np.zeros((4,4))

    for i in range(5):
        X,Y = feed.next_batch(batch_size)
        pred = net.forward(X).argmax(axis=1)
        
        targets = Y.argmax(axis=1)
        
        for i in range(4):
            for j in range(4):
                conf_mat[i,j] += ((pred==i)*(targets==j)).sum()

    print(conf_mat)
    print(np.diagonal(conf_mat).sum()*1./(conf_mat.sum()))


if __name__ == "__main__":
    train()  
    # test()  