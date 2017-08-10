import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
import matplotlib.image as mpimg
from scipy.ndimage import filters
from scipy.misc import imsave
import urllib
from numpy import random
import csv
import sys

import tensorflow as tf
import json
from datetime import datetime
from MITOS12Feed import MITOS12Feed

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
    def __init__(self, batch_size, im_size, sess, lr=0, eps=0, a=0):
        net_data = np.load("./alexnet/bvlc_alexnet.py35.npy").item()
        for key in net_data:
            print(key, net_data[key][0].shape, net_data[key][1].shape)

        self.x = tf.placeholder(tf.float32, [batch_size,im_size[0],im_size[1],3])
        res = tf.image.resize_images(self.x, (227, 227))

        #conv1 / kernel : (11, 11, 96) / stride : (4, 4) / name='conv1'
        conv1W = tf.Variable(tf.truncated_normal(net_data["conv1"][0].shape))
        conv1b = tf.Variable(tf.truncated_normal(net_data["conv1"][1].shape))
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
        conv2W = tf.Variable(tf.truncated_normal(net_data["conv2"][0].shape))
        conv2b = tf.Variable(tf.truncated_normal(net_data["conv2"][1].shape))
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
        conv3W = tf.Variable(tf.truncated_normal(net_data["conv3"][0].shape))
        conv3b = tf.Variable(tf.truncated_normal(net_data["conv3"][1].shape))
        conv3 = tf.nn.relu(conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1))

        #conv4 / (3, 3, 384) / (1, 1)
        conv4W = tf.Variable(tf.truncated_normal(net_data["conv4"][0].shape))
        conv4b = tf.Variable(tf.truncated_normal(net_data["conv4"][1].shape))
        conv4 = tf.nn.relu(conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2))

        #conv5
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv5W = tf.Variable(tf.truncated_normal(net_data["conv5"][0].shape))
        conv5b = tf.Variable(tf.truncated_normal(net_data["conv5"][1].shape))
        conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)

        #maxpool5
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        #fc6
        #fc(4096, name='fc6')
        fc6W = tf.Variable(tf.truncated_normal(net_data["fc6"][0].shape))
        fc6b = tf.Variable(tf.truncated_normal(net_data["fc6"][1].shape))
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [batch_size, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

        #fc7
        #fc(4096, name='fc7')
        fc7W = tf.Variable(tf.truncated_normal(net_data["fc7"][0].shape))
        fc7b = tf.Variable(tf.truncated_normal(net_data["fc7"][1].shape))
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

        #fc8
        #fc(2, relu=False, name='fc8')
        fc8W = tf.Variable(tf.truncated_normal([4096,2]))
        fc8b = tf.Variable(tf.truncated_normal([2]))
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

        # Softmax
        softmax = tf.nn.softmax(fc8)

        self.output = softmax

        # Setup Training
        self.train_W = [fc8W, fc7W, fc6W, conv5W, conv4W, conv3W, conv2W, conv1W]
        self.train_b = [fc8b, fc7b, fc6b, conv5b, conv4b, conv3b, conv2b, conv1b]
        self.train_vars = self.train_W+self.train_b
        self.target = tf.placeholder(tf.float32, [batch_size, 2])

        # Using cross-entropy
        self.loss = -tf.reduce_sum(self.target*tf.log(tf.clip_by_value(self.output,1e-10,1.0))) #-tf.reduce_mean(self.target*tf.log(self.output) + (1-self.target)*tf.log(1-self.output))
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.train_W])
        self.cost = self.loss+a*self.l2_loss
        optimizer = tf.train.AdamOptimizer(lr, epsilon=eps)
        self.trainingStep = optimizer.minimize(self.cost, var_list=self.train_vars)

        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def forward(self, X):
        return self.sess.run(self.output, feed_dict={self.x:X})

    def train(self, X, Y):
        self.trainingStep.run(feed_dict={self.x: X, self.target: Y})
    
    def get_cost(self, X, Y):
        return (self.loss.eval(feed_dict={self.x: X, self.target: Y}), self.l2_loss.eval(feed_dict={self.x: X, self.target: Y}))

def train():
    clf_name = "Mitos12_AlexNet_scratch"
    sess = tf.InteractiveSession()
    batch_size = 1
    feed = MITOS12Feed(batch_size=batch_size)
    tile_size = (127,127)
    lr = 1e-3
    eps = 0.1
    a = 0.0002
    net = AlexNet(batch_size, tile_size, sess, lr, eps, a)
    saver = tf.train.Saver(net.train_vars)
    best_score = 10000000000
    restore_from = "./Mitos12_AlexNet_scratch_201701181710_last.ckpt"

    now = datetime.now().strftime("%Y%m%d%H%M")
    clf_date_name = '%s_%s'%(clf_name, now)
    with open('%s.log'%clf_date_name, 'w', encoding="utf8") as logfile:
        run_info = {'clf_name': clf_name,
                    'batch_size': batch_size,
                    'tile_size': tile_size,
                    'description': "AlexNet, cut at level 4, added 1 maxpool, 2 fully connected relu6 and 1 softmax + fine-tuning of layer 3-4. MITOS12 Dataset with random deformations",
                    'file': "alexnet_mitos12.py",
                    'lr': lr,
                    'eps': eps,
                    'a': a,
                    'continuing_from_previous_run': False,
                    # 'restored_clf': restore_from,
                    'Date': now}
        json.dump(run_info, logfile, indent=4)
        logfile.write("\n")

    # saver.restore(sess, restore_from)
    # if(os.path.isdir('weights/%s'%clf_date_name) == False):
    #     os.mkdir('weights/%s'%clf_date_name)

    saver.save(sess, "./%s_first.ckpt"%clf_date_name)

    print("Start learning %s..."%clf_date_name)
    for i in range(1000001):
        if i%500==0:
            X,Y = feed.next_single_batch()
            cost = net.get_cost(X,Y)
            with open('%s_%s.log'%(clf_name, now), 'a', encoding="utf8") as logfile:
                logfile.write("Cost at iteration %d : %f + a*%f\n"%(i, cost[0], cost[1]))
            if cost[0]+a*cost[1] < best_score:
                print("Best score : %f."%(cost[0]+a*cost[1]))
                best_score = cost[0]+a*cost[1]
                saver.save(sess, "./%s_best.ckpt"%clf_date_name)
                # print(get_conf_mat(net,X,Y,2))

        X,Y = feed.next_single_batch()
        net.train(X,Y)

    saver.save(sess, "./%s_last.ckpt"%clf_date_name)

def get_conf_mat(net, X, Y, n_classes):
    conf_mat = np.zeros((n_classes,n_classes))

    pred = net.forward(X).argmax(axis=1)
    targets = Y.argmax(axis=1)
        
    for i in range(n_classes):
        for j in range(n_classes):
            conf_mat[i,j] += ((pred==i)*(targets==j)).sum()

    return conf_mat

def test():
    clf_name = "Mitos12_AlexNet_cut4"
    feed = MITOS12Feed()
    sess = tf.InteractiveSession()
    batch_size = 1
    tile_size = (127,127)
    lr = 1e-3
    eps = 1.0
    a = 0.001
    net = AlexNet(batch_size, tile_size, sess, lr, eps, a)
    saver = tf.train.Saver(net.train_vars)

    saver.restore(sess, "./Mitos12_AlexNet_scratch_201701181710_last.ckpt")

    conf_mat = np.zeros((2,2))

    for i in range(200):
        X,Y = feed.next_single_batch()
        pred = net.forward(X)
        print(pred[0], Y[0])

    '''for i in range(5):
        X,Y = feed.next_batch()
        conf_mat += get_conf_mat(net,X,Y,2)

    print(conf_mat)
    print(np.diagonal(conf_mat).sum()*1./(conf_mat.sum()))'''

def get_prob_mask():
    # Mask for adding probs: 1 at the center, decreasing to 0 at the corners
    P = np.zeros((127,127))
    for x in range(127):
        for y in range(127):
            P[y,x] = (y-63)**2+(x-63)**2
    P = (P.max()-P)/(P.max()-P.min())
    return P

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

def run_complete_test():
    testdir = 'e:/data/MITOS12/test'
    patchesdir = os.path.join(testdir, 'patches')
    clf = './Mitos12_AlexNet_cut4_201701171424_best.ckpt'

    res = 0.2456 #µm/px

    images = [f for f in os.listdir(testdir) if f.find('.bmp') >= 0]
    
    batch_size = 196
    tile_size = (127,127)
    lr = 1e-3
    eps = 1.0
    a = 0.001

    sess = tf.InteractiveSession()
    net = AlexNet(batch_size, tile_size, sess, lr, eps, a)
    saver = tf.train.Saver(net.train_vars)

    P = get_prob_mask()
    # thresh = 0.91 # Threshold determined from the training set
    saver.restore(sess, clf)
    print(images)

    for imfile in images:
        if( imfile == 'A00_00.bmp' ): continue
        print(imfile)
        patches = [f for f in os.listdir(patchesdir) if f.find(imfile) >= 0]
        classes = np.zeros((2084,2084,2)) # channels : max(class0), max(class1)
        csv = os.path.join(testdir, imfile.replace('bmp','csv'))
        GT = get_ground_truth(csv)
        for p in patches:
            y = int(p[11:15])*10
            patch = np.load(os.path.join(patchesdir,p)).astype('float')-127
            
            Y = net.forward(patch)
            for x in range(196):
                classes[y:y+127,x*10:(x*10)+127,0] = np.maximum(classes[y:y+127,x*10:(x*10)+127,0],Y[x,0]*P)
                classes[y:y+127,x*10:(x*10)+127,1] = np.maximum(classes[y:y+127,x*10:(x*10)+127,1],Y[x,1]*P)

            print(int(p[11:15]),'/',195)

        np.save('%s.%s.npy'%(clf,imfile), classes)
        #M = classes[:,:,1]>=thresh
        #true_centroids = [r.centroid for r in regionprops(label(GT))]
        #clf_centroids = [r.centroid for r in regionprops(label(M))]



def test_image():
    patchesdir = 'e:/data/MITOS12/test/patches'
    imfile = 'A00_00.bmp'
    imsdir = 'e:/data/MITOS12/test/'
    clf = './Mitos12_AlexNet_cut4_201701171424_best.ckpt'

    im = plt.imread(os.path.join(imsdir,imfile))
    classes = np.zeros(im.shape) # channels : max(class0), max(class1)
    sess = tf.InteractiveSession()

    batch_size = 196
    tile_size = (127,127)
    lr = 1e-3
    eps = 1.0
    a = 0.001
    net = AlexNet(batch_size, tile_size, sess, lr, eps, a)
    saver = tf.train.Saver(net.train_vars)

    # Mask for adding probs: 1 at the center, decreasing to 0 at the corners
    P = get_prob_mask()

    saver.restore(sess, clf)

    patches = [f for f in os.listdir(patchesdir) if f.find(imfile) >= 0]
    for p in patches:
        y = int(p[11:15])*10
        patch = np.load(os.path.join(patchesdir,p)).astype('float')-127
        
        Y = net.forward(patch)
        for x in range(196):
            classes[y:y+127,x*10:(x*10)+127,0] = np.maximum(classes[y:y+127,x*10:(x*10)+127,0],Y[x,0]*P)
            classes[y:y+127,x*10:(x*10)+127,1] = np.maximum(classes[y:y+127,x*10:(x*10)+127,1],Y[x,1]*P)

        print(int(p[11:15]),'/',195)

    np.save('%s.%s.npy'%(clf,imfile), classes)

if __name__ == "__main__":
    train()
    # test()
    # test_image()
    # run_complete_test()