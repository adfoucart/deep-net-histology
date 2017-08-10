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


def get_network(X):
    batch_size = X.get_shape()[0]
    im_size = tf.constant([X.get_shape().as_list()[1], X.get_shape().as_list()[2]])
    channels = X.get_shape()[3]
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0

    net = tf.image.resize_images(X, (227, 227))
    net = tf.contrib.layers.conv2d(net, 96, 11, 4, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='conv1')
    net = tf.nn.local_response_normalization(net, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name='lrn1')
    net = tf.contrib.layers.max_pool2d(net, 3, 2, 'SAME', scope='mp1')
    net = tf.contrib.layers.conv2d(net, 256, 5, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='conv2')
    net = tf.nn.local_response_normalization(net, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name='lrn2')
    net = tf.contrib.layers.max_pool2d(net, 3, 2, 'SAME', scope='mp2')
    net = tf.contrib.layers.conv2d(net, 384, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='conv3')
    net = tf.contrib.layers.conv2d(net, 384, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='conv4')
    net = tf.contrib.layers.conv2d(net, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='conv5')
    net = tf.contrib.layers.max_pool2d(net, 3, 2, 'SAME', scope='mp5')

    net = tf.contrib.layers.conv2d_transpose(net, 256, 5, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='tconv1')
    net = tf.contrib.layers.conv2d_transpose(net, 256, 5, 4, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='tconv2')
    net = tf.contrib.layers.conv2d_transpose(net, 64, 5, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='tconv3')
    net = tf.contrib.layers.conv2d_transpose(net, 1, 1, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='sig')
    net = tf.image.resize_images(net, im_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    '''net1 = tf.contrib.layers.conv2d(X, 32, 11, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='conv1_1')
    net2 = tf.contrib.layers.conv2d(X, 32, 7, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='conv1_2')
    net3 = tf.contrib.layers.conv2d(X, 32, 5, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='conv1_3')
    net4 = tf.image.resize_images(X, [64,64], method=tf.image.ResizeMethod.BILINEAR)#tf.contrib.layers.conv2d(X, 32, 1, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='conv1_4')
    net = tf.concat([net1, net2, net3, net4], 3)
    net = tf.contrib.layers.max_pool2d(net, 4, 1, 'SAME', scope='conv1_mp')
    
    net1 = tf.contrib.layers.conv2d(net, 64, 11, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='conv2_1')
    net2 = tf.contrib.layers.conv2d(net, 64, 7, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='conv2_2')
    net3 = tf.contrib.layers.conv2d(net, 64, 5, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='conv2_3')
    net4 = net4 = tf.image.resize_images(net, [32,32], method=tf.image.ResizeMethod.BILINEAR)#tf.contrib.layers.conv2d(net, 64, 1, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='conv2_4')
    net = tf.concat([net1, net2, net3, net4], 3)
    net = tf.contrib.layers.max_pool2d(net, 2, 1, 'SAME', scope='conv2_mp')

    net = tf.contrib.layers.conv2d(net, 128, 5, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='conv3')
    
    net = tf.contrib.layers.conv2d_transpose(net, 128, 5, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='tconv1')    
    net = tf.contrib.layers.conv2d_transpose(net, 64, 5, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='tconv2')    
    net = tf.contrib.layers.conv2d_transpose(net, 32, 5, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='tconv3')

    net = tf.contrib.layers.conv2d(net, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='conv_end')

    net = tf.image.resize_images(net, im_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    net = tf.sigmoid(net, name="sigmoid")'''

    return net

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


def test(saver, batches, net, sess, clf_name, loss):
    saver.restore(sess, "e:/data/tf_checkpoint/%s.ckpt"%clf_name)

    epoch = 1
    for batch in feed(batches, epoch):
        t = batch[1].astype('float')
        t[t==0] = -1.
        Y = net.eval(session=sess, feed_dict={X: batch[0], target: t})
        # loss = sess.run(loss, feed_dict={X: batch[0], target: t})
        #print(loss, np.sqrt((Y-t)**2).sum()/(20*127*127))
        plt.figure()
        for i in range(len(batch[0])):
            im = (batch[0][i] - batch[0][i].min())/(batch[0][i].max()-batch[0][i].min())
            plt.subplot(4,5,i+1)
            plt.imshow(im)
        plt.figure()
        for i in range(len(batch[1])):
            plt.subplot(4,5,i+1)
            plt.imshow(batch[1][i][:,:,0])
        plt.figure()
        for i in range(len(batch[1])):
            plt.subplot(4,5,i+1)
            plt.imshow(Y[i][:,:,0])
        plt.show()
        break

def train(saver, batches, net, sess, clf_name, loss, merged, train_writer, restore_from=None):
    c = 10000
    i = 0
    n_stable = 0

    if restore_from != None:
        saver.restore(sess, "e:/data/tf_checkpoint/%s.ckpt"%restore_from)

    for epoch in range(10):
        for batch in feed(batches, epoch):
            t = batch[1].astype('float')
            t[t==0] = -1.
            t[t>0] = 1.
            trainingStep.run(session=sess, feed_dict={X: batch[0], target: t})
            if( i % 100 == 0 ):
                [summ, lv] = sess.run([merged,loss], feed_dict={X: batch[0], target: t})
                train_writer.add_summary(summ, i)
                # if( np.abs(lv-c) < 1e-4 ):
                #     n_stable += 1
                # if lv < c : 
                #     c = lv
                # else:
                #     n_stable = 0
                # if n_stable >= 10:
                #     print("Converged after %d iterations."%i)
                #     break
                saver.save(sess, "e:/data/tf_checkpoint/%s.ckpt"%clf_name)
            i += 1

        saver.save(sess, "e:/data/tf_checkpoint/%s.ckpt"%clf_name)
        if n_stable >= 10:
            break

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

def run_complete_test(saver, net, sess, clf_name):
    testdir = 'e:/data/MITOS12/test'
    patchesdir = os.path.join(testdir, 'patches')
    clf = "e:/data/tf_checkpoint/%s.ckpt"%clf_name

    res = 0.2456 #µm/px

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
            
            Y = net.eval(session=sess, feed_dict={X: patch})
            for x in range(196):
                classes[y:y+127,x*10:(x*10)+127] += Y[x,:,:,0]*P
                ns[y:y+127,x*10:(x*10)+127] += P

            print(int(p[11:15]),'/',195)

        classes[ns>0.5] /= ns[ns>0.5]
        classes[ns<= 0.5] = 0
        np.save('%s.%s.npy'%(clf,imfile), classes)
        #M = classes[:,:,1]>=thresh
        #true_centroids = [r.centroid for r in regionprops(label(GT))]
        #clf_centroids = [r.centroid for r in regionprops(label(M))]


if __name__ == "__main__":
    batch_size = 196#20             # Use 196 for final test, 20 for training
    im_size = (127,127)
    lr = 1e-2
    eps = 0.1
    a = 0.

    clf_name = "fcnet_pretrained"
    clf_from = "fcnet_scratch"

    # batches = "E:\\data\\MITOS12\\train\\batches_with_stuff"
    batches = "E:\\data\\MITOS12\\train\\batches"

    X = tf.placeholder(tf.float32, [batch_size,im_size[0],im_size[1],3])

    net = get_network(X)
    
    saver = tf.train.Saver()

    target = tf.placeholder(tf.float32, [batch_size, im_size[0],im_size[1],1])
    loss = tf.losses.mean_squared_error(target, net)#tf.sqrt(tf.reduce_mean(tf.square(tf.sub(target,net))))
    loss = tf.add_n([loss] + [a*r for r in tf.losses.get_regularization_losses()])
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()

    optimizer = tf.train.AdamOptimizer(lr, epsilon=eps)
    trainingStep = optimizer.minimize(loss)

    sess = tf.Session()
    train_writer = tf.summary.FileWriter('./summaries/train', sess.graph)
    sess.run(tf.global_variables_initializer())

    sumim = tf.summary.image("X", X, max_outputs=3)
    sumtar = tf.summary.image("Target", target, max_outputs=3)
    
    tf.get_default_graph().finalize()

    # test(saver, batches, net, sess, clf_name, loss)
    # train(saver, batches, net, sess, clf_name, loss, merged, train_writer, clf_from)
    run_complete_test(saver, net, sess, clf_name)


    # test()
    # test_image()
    # run_complete_test()