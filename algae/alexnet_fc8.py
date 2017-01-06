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
        net_data = np.load("../alexnet/bvlc_alexnet.npy").item()

        self.x = tf.placeholder(tf.float32, [batch_size,im_size[0],im_size[1],3])
        res = tf.image.resize_images(self.x, 227, 227)

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

        #conv5 / (3, 3, 256) / (1, 1)
        conv5W = tf.Variable(net_data["conv5"][0])
        conv5b = tf.Variable(net_data["conv5"][1])
        conv5 = tf.nn.relu(conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding="SAME", group=2))

        #maxpool5
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        #fc6 (4096)
        fc6W = tf.Variable(net_data["fc6"][0])
        fc6b = tf.Variable(net_data["fc6"][1])
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [batch_size, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

        #fc7 (4096)
        fc7W = tf.Variable(net_data["fc7"][0])
        fc7b = tf.Variable(net_data["fc7"][1])
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

        #fc8 (1000)
        fc8W = tf.Variable(net_data["fc8"][0])
        fc8b = tf.Variable(net_data["fc8"][1])
        fc8 = tf.nn.relu_layer(fc7, fc8W, fc8b)

        #fc9 (4)
        fc9W = tf.Variable(tf.truncated_normal([1000,4], stddev=0.1))
        fc9b = tf.Variable(tf.zeros([4]))
        fc9 = tf.nn.relu_layer(fc8, fc9W, fc9b)

        # Softmax
        softmax = tf.nn.softmax(fc9)

        self.output = softmax
        self.features = fc8

        # Setup Training
        self.train_W = [fc9W, fc8W, fc7W, fc6W]#, conv5W, conv4W, conv3W, conv2W, conv1W]
        self.train_b = [fc9b, fc8b, fc7b, fc6b]#, conv5b, conv4b, conv3b, conv2b, conv1b]
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


'''class DataFeed:

    def __init__(self, dataset_path, patches_path, verbose=True):
        seed = 1

        self.v = verbose
        if self.v: print "Loading dataset and patches"
        self.dataset = [os.path.join(dataset_path,f) for f in os.listdir(dataset_path) if f.find('.npy') > 0]
        self.patches = [os.path.join(patches_path,f) for f in os.listdir(patches_path) if f.find('.npy') > 0]
        if len(self.dataset) != len(self.patches):
            raise Exception("Dataset and patches should have the same number of files")

        # Get number of patches
        n_patches = []
        for f in self.dataset:
            a = np.load(f)
            n_patches += [a.shape[0]]
        n_patches = np.array(n_patches)
        if self.v: print "Total number of patches : %d"%n_patches.sum()

        file_ids = np.arange(len(self.dataset))

        if self.v: print "Using seed : %d"%seed
        np.random.seed(seed) # Seed used to always use the same training set / test set for future reference
        np.random.shuffle(file_ids)

        # Find cutoff so that at most 80% of patches are in the training set
        cut_patches = int(n_patches.sum()*0.8)
        if self.v: print "Trying to get at most %d patches in training set"%cut_patches
        t = 0
        self.training_set = []
        self.test_set = []
        for i in file_ids:
            t += n_patches[i]
            if t <= cut_patches:
                self.training_set += [i]
            else:
                self.test_set += [i]

        self.training_set = np.array(self.training_set)
        self.test_set = np.array(self.test_set)
        if self.v: print "Patches in training set : %d"%n_patches[self.training_set].sum()
        if self.v: print "Patches in test set : %d"%n_patches[self.test_set].sum()

        self.next_file = 0
        self.current_patch = None
        self.current_dataset = None
        self.pos_in_patch = 0

    def load_next_file(self):
        if self.v: print "Loading patches & dataset from next tile"
        i = self.training_set[self.next_file]
        try:
            self.current_patch = np.load(self.patches[i])
            self.current_dataset = np.load(self.dataset[i])
        except ValueError as e:
            print "i:",i
            print "self.next_file:",self.next_file
            print self.patches[i]

        self.next_file = (self.next_file+1)%len(self.training_set)
        self.pos_in_patch = 0

    def next_batch(self,batch_size,add_noise=False,add_fuzzy=False):
        if self.current_patch is None:
            self.load_next_file()

        if( self.current_patch.shape[0]-batch_size-self.pos_in_patch < 0 ):
            if self.v: print "Between two files..."
            x0 = self.current_patch[self.pos_in_patch:]
            y0 = self.current_dataset[self.pos_in_patch:,4:]
            self.load_next_file()
            x1 = self.current_patch[:batch_size-x0.shape[0]]
            y1 = self.current_dataset[:batch_size-x0.shape[0],4:]
            self.pos_in_patch = batch_size-x0.shape[0]

            X = np.vstack([x0,x1])
            Y = np.vstack([y0,y1])
            if( X.shape[0] != batch_size ): return self.next_batch(batch_size)
            if add_noise : X = self.add_noise(X)
            if add_fuzzy : Y = self.add_fuzzy(Y)
            return X.astype('float32')-127,Y

        X = self.current_patch[self.pos_in_patch:self.pos_in_patch+batch_size]
        Y = self.current_dataset[self.pos_in_patch:self.pos_in_patch+batch_size,4:]
        self.pos_in_patch += batch_size
        
        if( X.shape[0] != batch_size ): return self.next_batch(batch_size)
        
        if add_noise : X = self.add_noise(X)
        if add_fuzzy : Y = self.add_fuzzy(Y)
        return X.astype('float32')-127,Y'''


def train():
    clf_name = "AlexNet_8finetune_1fc_1softmax"
    feed = EqualizedDataFeed("/mnt/e/data/algae_dataset_equal_batches", False)
    sess = tf.InteractiveSession()
    batch_size = 200
    tile_size = (128,128)
    lr = 1e-3
    eps = 0.01
    a = 0.001
    net = AlexNet(batch_size, tile_size, sess, lr, eps, a)
    saver = tf.train.Saver(net.train_vars)
    best_score = 5.

    now = datetime.now().strftime("%Y%m%d%H%M")
    clf_date_name = '%s_%s'%(clf_name, now)
    with open('%s.log'%clf_date_name, 'wb') as logfile:
        run_info = {'clf_name': clf_name,
                    'batch_size': batch_size,
                    'tile_size': tile_size,
                    'description': "AlexNet, fine-tuned level 6-8, added 1 fully connected + 1 softmax.",
                    'file': "alexnet_fc8.py",
                    'lr': lr,
                    'eps': eps,
                    'a': a,
                    'continuing_from_previous_run': True,
                    'restored_clf': "AlexNet_8finetune_1fc_1softmax_201611031001_last.ckpt",
                    'Date': now}
        json.dump(run_info, logfile, indent=4)
        logfile.write("\n")

    saver.restore(sess, "AlexNet_8finetune_1fc_1softmax_201611031001_last.ckpt")
    # if(os.path.isdir('weights/%s'%clf_date_name) == False):
    #     os.mkdir('weights/%s'%clf_date_name)

    print "Start learning %s..."%clf_date_name
    for i in range(20001):
        if i%10==0:
            X,Y = feed.next_batch(batch_size)
            cost = net.get_cost(X,Y)
            with open('%s_%s.log'%(clf_name, now), 'a') as logfile:
                logfile.write("Cost at iteration %d : %f + a*%f\n"%(i, cost[0], cost[1]))
            if cost[0] < best_score:
                print "Best score : %f."%cost[0]
                best_score = cost[0]
                saver.save(sess, "%s_best.ckpt"%clf_date_name)
                # np.save('weights/%s/wtop_best.npy'%clf_date_name, net.train_W[0].eval())
                # np.save('weights/%s/wbottom_best.npy'%clf_date_name, net.train_W[-1].eval())
            # np.save('weights/%s/wtop_%05d.npy'%(clf_date_name,i), net.train_W[0].eval())

        X,Y = feed.next_batch(batch_size)
        net.train(X,Y)

    saver.save(sess, "%s_last.ckpt"%clf_date_name)

def test():
    clf_name = "AlexNet_8finetune_1fc_1softmax"
    # feed = DataFeed("/mnt/e/data/algae_dataset", "/mnt/e/data/algae_patches", False)
    # feed = DataFeed("/mnt/e/data/algae_dataset_cells_only", "/mnt/e/data/algae_patches_cells_only", False, False)
    feed = EqualizedDataFeed("/mnt/e/data/algae_dataset_equal_batches", False)
    sess = tf.InteractiveSession()
    batch_size = 200
    tile_size = (128,128)
    lr = 1e-3
    eps = 1.0
    a = 0.001
    net = AlexNet(batch_size, tile_size, sess, lr, eps, a)
    saver = tf.train.Saver(net.train_vars)

    # saver.restore(sess, "AlexNet_8finetune_1fc_1softmax_201611031001_best.ckpt")

    conf_mat = np.zeros((4,4))

    for i in range(5):
        X,Y = feed.next_batch(batch_size)
        pred = net.forward(X).argmax(axis=1)
        
        targets = Y.argmax(axis=1)
        
        for i in range(4):
            for j in range(4):
                conf_mat[i,j] += ((pred==i)*(targets==j)).sum()

    print conf_mat
    print np.diagonal(conf_mat).sum()*1./(conf_mat.sum())

def features_viz():
    clf_name = "AlexNet_8finetune_1fc_1softmax"
    # feed = DataFeed("/mnt/e/data/algae_dataset", "/mnt/e/data/algae_patches", False)
    # feed = DataFeed("/mnt/e/data/algae_dataset_cells_only", "/mnt/e/data/algae_patches_cells_only", False, False)
    feed = EqualizedDataFeed("/mnt/e/data/algae_dataset_equal_batches", False)
    sess = tf.InteractiveSession()
    batch_size = 200
    tile_size = (128,128)
    lr = 1e-3
    eps = 1.0
    a = 0.001
    net = AlexNet(batch_size, tile_size, sess, lr, eps, a)
    saver = tf.train.Saver(net.train_vars)

    saver.restore(sess, "AlexNet_8finetune_1fc_1softmax_201611031001_best.ckpt")
    feature_mat = np.zeros((10000,1000))

    for i in range(50):
        X,Y = feed.next_batch(batch_size)
        feature_mat[i*200:(i+1)*200,:] = net.features.eval(feed_dict={net.x:X})

    np.save("AlexNet_8finetune_1fc_1softmax_201611031001_best_features.ckpt.npy", feature_mat)

if __name__ == "__main__":
    # train()
    # test()
    features_viz()

'''if __name__ == "__main__":
    clf_name = "AlexNet_8finetune_1fc_1softmax"
    feed = DataFeed("/mnt/e/data/algae_dataset_cells_only", "/mnt/e/data/algae_patches_cells_only", False)

    sess = tf.InteractiveSession()
    batch_size = 10
    tile_size = (128,128)
    lr = 1e-3
    eps = 1.0
    a = 0.0
    net = AlexNet(batch_size, tile_size, sess, lr, eps, a)


    saver = tf.train.Saver(net.train_vars)
    best_score = 10000

    now = datetime.now().strftime("%Y%m%d%H%M")
    with open('%s_%s.log'%(clf_name, now), 'wb') as logfile:
        run_info = {'clf_name': clf_name,
                    'batch_size': batch_size,
                    'tile_size': tile_size,
                    'description': "AlexNet, fine-tuned level 6-8, added 1 fully connected + 1 softmax.",
                    'file': "alexnet_fc8.py",
                    'lr': lr,
                    'eps': eps,
                    'a': a,
                    'continuing_from_previous_run': False,
                    'Date': now}
        json.dump(run_info, logfile, indent=4)
        logfile.write("\n")

    # saver.restore(sess, "%s.ckpt"%clf_name)

    print "Start learning..."
    for i in range(10001):
        if i%10==0:
            X,Y = feed.next_batch(batch_size)
            cost = net.get_cost(X,Y)
            with open('%s_%s.log'%(clf_name, now), 'a') as logfile:
                logfile.write("Cost at iteration %d : %f + a*%f\n"%(i, cost[0], cost[1]))
            if cost[0] < best_score:
                print "Best score : %f."%cost[0]
                best_score = cost[0]
                saver.save(sess, "%s.ckpt"%clf_name)
                np.save('weights/wtop_best.npy', net.train_W[0].eval())
                np.save('weights/wbottom_best.npy', net.train_W[-1].eval())
            np.save('weights/wtop_%05d.npy'%i, net.train_W[0].eval())

        X,Y = feed.next_batch(batch_size)
        net.train(X,Y)'''
        