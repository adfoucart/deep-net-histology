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

'''
slide_defect_resnet_logits5
    # Widen the network
    net1 = tf.contrib.layers.conv2d(X, 256, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    res1 = add_residual(net1, 'res1', True, 128)
    res2 = add_residual(res1, 'res2', True, 128)
    res3 = add_residual(res2, 'res3', True, 128)
    up1 = add_up(res3, 'up1')
    res4 = add_residual(up1, 'res4', False, 128)
    up2 = add_up(res4, 'up2')
    res5 = add_residual(up2, 'res5', False, 128)
    up3 = add_up(res5, 'up3')

    seg1 = tf.contrib.layers.conv2d(up3, 64, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='seg/1')
    net = tf.contrib.layers.conv2d(seg1, 2, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='seg/2')

    return [net]

NOTE : resnet_c is the first for slide_defect to include data augmentation. Dataset has been improved starting from slide_defect_resnet_7
'''

def add_residual(net_in, name, with_mp=True, width=128):
    neta = tf.contrib.layers.conv2d(net_in, width, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='%s/1'%name)
    netb = tf.contrib.layers.conv2d(neta, width, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='%s/2'%name)
    netc = tf.contrib.layers.conv2d(netb, net_in.get_shape()[3], 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='%s/3'%name)
    net = tf.add(net_in, netc, name='%s/add'%name)
    if with_mp:
        net = tf.contrib.layers.max_pool2d(net, 3, 2, 'SAME', scope='%s/mp'%name)
    return net

def add_up(net_in, name, width=None):
    if width==None:
        return tf.contrib.layers.conv2d_transpose(net_in, net_in.get_shape()[3], 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='%s'%name)
    else:
        return tf.contrib.layers.conv2d_transpose(net_in, width, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='%s'%name)

def add_seg(net_in, name, n_inter=64):
    net = tf.contrib.layers.conv2d(net_in, n_inter, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='%s/1'%name)
    net = tf.contrib.layers.conv2d(net, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='%s/2'%name)
    return net

def get_network(X, kp):
    batch_size = X.get_shape()[0]
    im_size = tf.constant([X.get_shape().as_list()[1], X.get_shape().as_list()[2]])
    channels = X.get_shape()[3]

    # Widen the network
    net1 = tf.contrib.layers.conv2d(X, 256, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    res1 = add_residual(net1, 'res1', True, 128)
    res2 = add_residual(res1, 'res2', True, 128)
    res3 = add_residual(res2, 'res3', True, 128)
    up1 = add_up(res3, 'up1')
    res4 = add_residual(up1, 'res4', False, 128)
    up2 = add_up(res4, 'up2')
    res5 = add_residual(up2, 'res5', False, 128)
    up3 = add_up(res5, 'up3')

    res6 = add_residual(up3, 'res6', False, 128)
    res7 = add_residual(res6, 'res7', False, 128)

    seg1 = tf.contrib.layers.conv2d(res7, 64, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='seg/1')
    net = tf.contrib.layers.conv2d(seg1, 2, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='seg/2')

    return [net]

def transform_batch(X,Y,isTraining):
    Y2 = np.zeros((Y.shape[0], Y.shape[1], Y.shape[2], 2))
    Y2[:,:,:,0] = Y[:,:,:,0]
    Y2[:,:,:,1] = 1-Y[:,:,:,0]

    if not isTraining:
        return X,Y2.reshape((Y2.shape[0]*Y2.shape[1]*Y2.shape[2],Y2.shape[3]))

    params = np.random.random((X.shape[0], 3))
    params[:,0] = np.floor(params[:,0]*4)
    params[:,1] *= 0.2
    params[:,2] = (params[:,2]-0.5)*0.2

    X2 = X.copy()

    # Orientation
    do_vswap = (params[:,0]==1)+(params[:,0]==3)
    do_hswap = (params[:,0]==2)+(params[:,0]==3)
    X2[do_vswap] = X2[do_vswap,::-1,:,:]
    X2[do_hswap] = X2[do_hswap,:,::-1,:]
    Y2[do_vswap] = Y2[do_vswap,::-1,:,:]
    Y2[do_hswap] = Y2[do_hswap,:,::-1,:]

    # Noise & illumination
    for i in range(X2.shape[0]):
        X2[i] += np.random.random(X2[i].shape)*params[i,1]-params[i,1]/2+params[i,2]

    return X2,Y2.reshape((Y2.shape[0]*Y2.shape[1]*Y2.shape[2],Y2.shape[3]))

def feed(path, seed=0, batch_size=20, isTraining=True):
    np.random.seed(seed)

    files = os.listdir(path)
    segfiles = [f for f in files if f.find('_seg.npy') > 0]
    basefnames = [f.replace('_seg.npy','') for f in segfiles]

    # Preload all coordinates
    coordinates = {}
    for b in basefnames:
        coordinates[b] = np.load(os.path.join(path,'%s_coordinates.npy'%b)).astype('int')

    tile_size = coordinates[b][0,2]-coordinates[b][0,0]

    N = len(basefnames)
    idxs = np.arange(N)
    np.random.shuffle(idxs)

    for idx in idxs:
        b = basefnames[idx]
        idc = np.arange(coordinates[b][coordinates[b][:,4]==1].shape[0])
        im = plt.imread(os.path.join(path,'%s.png'%b))[:,:,:3]
        seg = np.load(os.path.join(path,'%s_seg.npy'%b))
        np.random.shuffle(idc)
        coords = coordinates[b][coordinates[b][:,4]==1][idc[:batch_size]]
        X = np.zeros((batch_size,tile_size,tile_size,3))
        Y = np.zeros((batch_size,tile_size,tile_size,1))
        for i,c in enumerate(coords):
            X[i] = im[c[1]:c[1]+tile_size,c[0]:c[0]+tile_size,:]
            Y[i,:,:,0] = seg[c[1]:c[1]+tile_size, c[0]:c[0]+tile_size]
        yield transform_batch(X,Y,isTraining)

imscale = lambda X : (X-X.min())/(X.max()-X.min()) 

def test(saver, batches, net, sess, clf_name, loss, batch_size=20):
    saver.restore(sess, "e:/data/tf_checkpoint/%s.ckpt"%clf_name)

    epoch = 40
    for batch in feed(batches, epoch, batch_size, False):
        t = batch[1].astype('float')
        Y = net.eval(session=sess, feed_dict={X: batch[0], target: t})
        t = t.reshape((batch_size,128,128,2))
        plt.figure()
        for i in range(min(batch_size,10)):
            im = imscale(batch[0][i])
            plt.subplot(2,5,i+1)
            plt.imshow(im)
        plt.figure()
        for i in range(min(batch_size,10)):
            plt.subplot(2,5,i+1)
            plt.imshow(t[i][:,:,0])
        plt.figure()
        for i in range(min(batch_size,10)):
            plt.subplot(2,5,i+1)
            plt.imshow(Y[i][:,:,0])
            plt.colorbar()
        plt.figure()
        for i in range(min(batch_size,10)):
            plt.subplot(2,5,i+1)
            plt.imshow(Y[i][:,:,0]>0.5)
        plt.show()
        break

def train(saver, batches, net, sess, clf_name, loss, merged, train_writer, restore_from=None, batch_size=20):
    c = 5000
    i = 0

    print("Start training")

    if restore_from != None:
        saver.restore(sess, "e:/data/tf_checkpoint/%s.ckpt"%restore_from)

    for epoch in range(c):
        print("Epoch %d"%(epoch+1))
        for batch in feed(batches, epoch, batch_size, True):
            t = batch[1].astype('float')
            trainingStep.run(session=sess, feed_dict={X: batch[0], target: t})
            if( i % 100 == 0 ):
                [summ, lv] = sess.run([merged,loss], feed_dict={X: batch[0], target: t})
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

def detailed_test(saver, sess, net, clf_name, path):
    basenames = ['17H1743018_Scan 3', '17H1743018_Scan 1', '16H1779218_Scan 3']
    for test_basename in basenames:
    # test_basename = '17H1743018_Scan 3'#'17H1743018_Scan 1'#'16H1779218_Scan 3'#'17H1743018_Scan 3'
        fname_seg = os.path.join(path,'test/%s_seg.png'%test_basename)
        fname_or = os.path.join(path,'test/%s.png'%test_basename)
        clf = "e:/data/tf_checkpoint/%s.ckpt"%clf_name

        tile_size = 128
        stride = 64

        #color = np.array([182,255,0])/255. # segmentation color
        #im_seg = plt.imread(fname_seg)
        #b = np.abs(im_seg-color).sum(axis=2)<0.1 # Binary segmented image
        im = plt.imread(fname_or)[:,:,:3]

        imshape = im.shape
        nr,nc = (imshape[0]-tile_size)//stride,(imshape[1]-tile_size)//stride

        yr,xr = np.arange(1,nr)*stride,np.arange(1,nc)*stride
        mesh = np.meshgrid(xr,yr)
        tiles = zip(mesh[0].flatten(), mesh[1].flatten())

        patch_X = np.zeros((1,128,128,3))
        #patch_Y = np.zeros((1,128,128,1))
        result = np.zeros((imshape[0],imshape[1]))

        saver.restore(sess, clf)

        for t in tiles:
            patch_X[0] = im[t[1]:t[1]+tile_size, t[0]:t[0]+tile_size]
            #patch_Y[0,:,:,0] = b[t[1]:t[1]+tile_size, t[0]:t[0]+tile_size]
            
            Y = net.eval(session=sess, feed_dict={X: patch_X})
            result[t[1]+stride//2:t[1]+tile_size-stride//2, t[0]+stride//2:t[0]+tile_size-stride//2] = Y[0,stride//2:tile_size-stride//2,stride//2:tile_size-stride//2,0]

        np.save(os.path.join(path, 'results/result_%s_%s.npy'%(clf_name,test_basename)), result)

import sys
import datetime
if __name__ == "__main__":
    # Get command line argument : train / test / detailed / fulltest

    if( len(sys.argv) > 1 ):
        action = sys.argv[1]
        batch_size = 28 if action == 'fulltest' else 1 if action=='detailed' else 20            # Use 28 for final test, 20 for training
        im_size = (128,128)
        lr = 5e-5
        eps = 0.1
        a = 1e-6
        kp = 0.5 if action == 'train' else 1.

        clf_name = "slide_defect_logits" if len(sys.argv) <= 2 else sys.argv[2]
        clf_from = None if len(sys.argv) <= 3 else sys.argv[3] 

        batches = "E:\\data\\201702-scans\\images"

        X = tf.placeholder(tf.float32, [batch_size,im_size[0],im_size[1],3])

        #net,net6,net5,net5c,net4,net4c,net3,net3c,net2,net2c,net1 = get_network(X)
        nets = get_network(X, kp)
        net = nets[0]

        saver = tf.train.Saver()

        target = tf.placeholder(tf.float32, [batch_size*im_size[0]*im_size[1],2])
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=net))
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
            detailed_test(saver, sess, net, clf_name, batches)
        elif action == 'fulltest':
            run_complete_test(saver, net, sess, clf_name, True)
        else:
            print("Unknow command.")
            print("Usage : python resnet.py train|test|detailed|fulltest [clf_name] [clf_from]")

    else:
        print("Usage : python resnet.py train|test|detailed|fulltest [clf_name] [clf_from]")
