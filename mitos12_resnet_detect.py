import os
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy import random
import csv

import tensorflow as tf
import json
from datetime import datetime
# from MITOS12Feed import MITOS12Feed

'''
mitos12_resnet_detect6
    # Widen the network
    net1 = tf.contrib.layers.conv2d(X, 256, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    # Residual unit 1
    net2a = tf.contrib.layers.conv2d(net1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/1')
    net2b = tf.contrib.layers.conv2d(net2a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/2')
    net2c = tf.contrib.layers.conv2d(net2b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/3')
    net2 = tf.add(net1, net2c, name='res1/add')
    net2mp = tf.contrib.layers.max_pool2d(net2, 3, 2, 'SAME', scope='res1/mp')
    
    # Residual unit 2
    net3a = tf.contrib.layers.conv2d(net2mp, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/1')
    net3b = tf.contrib.layers.conv2d(net3a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/2')
    net3c = tf.contrib.layers.conv2d(net3b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/3')
    net3 = tf.add(net2mp, net3c, name='res2/add')
    net3mp = tf.contrib.layers.max_pool2d(net3, 3, 2, 'SAME', scope='res2/mp')
    
    # Residual unit 3
    net4a = tf.contrib.layers.conv2d(net3mp, 128, 3, 1, activation_fn=tf.nn.relu,  weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/1')
    net4b = tf.contrib.layers.conv2d(net4a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/2')
    net4c = tf.contrib.layers.conv2d(net4b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/3')
    net4 = tf.add(net3mp, net4c, name='res3/add')
    net4mp = tf.contrib.layers.max_pool2d(net4, 3, 2, 'SAME', scope='res3/mp')
    
    reshape = tf.reshape(net4mp, [batch_size, int(np.prod(net4mp.get_shape()[1:]))], name="reshape")

    # Fully connected layers
    fc5 = tf.contrib.layers.fully_connected(reshape, 64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.zeros_initializer(), scope='fc5')
    net = tf.contrib.layers.fully_connected(fc5, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.zeros_initializer(), scope='fc6')

    return [net]
'''


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

def add_residual(net_in, name, with_mp=True, width=128):
    neta = tf.contrib.layers.conv2d(net_in, width, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='%s/1'%name, padding='SAME' )
    netb = tf.contrib.layers.conv2d(neta, width, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='%s/2'%name, padding='SAME')
    netc = tf.contrib.layers.conv2d(netb, net_in.get_shape()[3], 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='%s/3'%name, padding='SAME')
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

    # Residual unit 1
    net2a = tf.contrib.layers.conv2d(net1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/1')
    net2b = tf.contrib.layers.conv2d(net2a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/2')
    net2c = tf.contrib.layers.conv2d(net2b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/3')
    net2 = tf.add(net1, net2c, name='res1/add')
    net2mp = tf.contrib.layers.max_pool2d(net2, 3, 2, 'SAME', scope='res1/mp')
    
    # Residual unit 2
    net3a = tf.contrib.layers.conv2d(net2mp, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/1')
    net3b = tf.contrib.layers.conv2d(net3a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/2')
    net3c = tf.contrib.layers.conv2d(net3b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/3')
    net3 = tf.add(net2mp, net3c, name='res2/add')
    net3mp = tf.contrib.layers.max_pool2d(net3, 3, 2, 'SAME', scope='res2/mp')
    
    # Residual unit 3
    net4a = tf.contrib.layers.conv2d(net3mp, 128, 3, 1, activation_fn=tf.nn.relu,  weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/1')
    net4b = tf.contrib.layers.conv2d(net4a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/2')
    net4c = tf.contrib.layers.conv2d(net4b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/3')
    net4 = tf.add(net3mp, net4c, name='res3/add')
    net4mp = tf.contrib.layers.max_pool2d(net4, 3, 2, 'SAME', scope='res3/mp')
    
    flat = tf.reshape(net4mp, [batch_size.value, int(np.prod(net4mp.get_shape()[1:]))], name="reshape")

    # Fully connected layers
    fc5 = tf.contrib.layers.fully_connected(flat, 64, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.zeros_initializer(), scope='fc5')
    net = tf.contrib.layers.fully_connected(fc5, 2, activation_fn=tf.nn.softmax, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.zeros_initializer(), scope='fc6')

    return [net]

max_mitosis_pixels = 40.
def transform_batch(X,Y,isTraining):
    p = np.minimum(Y/max_mitosis_pixels,1.) # prob = 1 if at least 40 pixels in the image are mitosis, 0 if 0 pixels, with linear progression in between
    Y2 = np.zeros((Y.shape[0], 2))
    Y2[:,0] = p
    Y2[:,1] = 1-p

    if not isTraining:
        return X,Y2

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

    # Noise & illumination
    for i in range(X2.shape[0]):
        X2[i] += np.random.random(X2[i].shape)*params[i,1]-params[i,1]/2+params[i,2]

    return X2,Y2

def get_files_from_path(path):
    path_x = os.path.join(path,'tiles')
    path_y = os.path.join(path,'targets')
    files = os.listdir(path_x)
    files_x = [os.path.join(path_x,f) for f in files]
    files_y = [os.path.join(path_y,f) for f in files]

    return files_x,files_y

def feed(path, seed=0, batch_size=20, isTraining=True):
    if(isinstance(path,list)):
        files_x = []
        files_y = []
        for p in path:
            fx,fy = get_files_from_path(p)
            files_x += fx
            files_y += fy
    else:
        files_x,files_y = get_files_from_path(path)

    np.random.seed(seed)

    #path_x = os.path.join(path,'tiles')
    #path_y = os.path.join(path,'targets')

    #files = os.listdir(path_x)
    
    tile_size = 128

    N = len(files_x)
    idxs = np.arange(N)
    np.random.shuffle(idxs)

    for idx in idxs:
        batch_x = np.load(files_x[idx])
        batch_y = np.load(files_y[idx])
        
        idc = np.arange(batch_y.shape[0])
        np.random.shuffle(idc)
        
        X = batch_x[idc[:batch_size]]
        #Y = batch_y[idc[:batch_size],:,:].sum(axis=1).sum(axis=1)
        Y = batch_y[idc[:batch_size],tile_size//2-tile_size//4:tile_size//2+tile_size//4,tile_size//2-tile_size//4:tile_size//2+tile_size//4].sum(axis=1).sum(axis=1)
        X,Y = transform_batch(X,Y,isTraining)
        yield X,Y,batch_x[idc[:batch_size]],batch_y[idc[:batch_size],:,:]

imscale = lambda X : (X-X.min())/(X.max()-X.min()) 

def test(saver, batches, net, sess, clf_name, loss, batch_size=20):
    wm = 0
    nm = 0
    i = 0
    epoch = 40
    for batch in feed(batches, epoch, batch_size, False):
        if( batch[1][:,0].sum() >= 0 ): nm += 1
        else: wm += 1
        i += 1
        if( i % 100 == 0 ): print(i,wm,nm)
        if( i > 1000 ): break
    print(i,wm,nm)
    '''saver.restore(sess, "e:/data/tf_checkpoint/%s.ckpt"%clf_name)

    epoch = 40
    for batch in feed(batches, epoch, batch_size, False):
        t = batch[1].astype('float')
        if( t[:,0].sum() <= 0 ): continue
        Y = net.eval(session=sess, feed_dict={X: batch[0], target: t})
        #loss = sess.run(loss, feed_dict={X: batch[0], target: t})
        #print(loss, np.sqrt((Y-t)**2).sum()/(batch_size*128*128))
        print(t)
        print(Y)
        plt.figure()
        for i in range(min(batch_size,10)):
            im = imscale(batch[0][i])
            plt.subplot(2,5,i+1)
            plt.imshow(im)
            plt.title('Target %d / Prediction %.2f'%(t[i,0], Y[i,0]))
        plt.figure()
        for i in range(min(batch_size,10)):
            im = imscale(batch[2][i])
            plt.subplot(2,5,i+1)
            plt.imshow(im)
        plt.figure()
        for i in range(min(batch_size,10)):
            im = imscale(batch[3][i])
            plt.subplot(2,5,i+1)
            plt.imshow(im)
        plt.show()
        break'''

def train(saver, batches, net, sess, clf_name, loss, merged, train_writer, restore_from=None, batch_size=20):
    c = 5000
    i = 0

    print("Start training")

    Xval = np.load("E:\\data\\MITOS12\\validation\\tiles\\500.npy")[:batch_size]
    Yval = np.load("E:\\data\\MITOS12\\validation\\targets\\500.npy")[:batch_size,:,:].sum(axis=1).sum(axis=1)
    Xval, Yval = transform_batch(Xval, Yval, False)

    if restore_from != None:
        saver.restore(sess, "e:/data/tf_checkpoint/%s.ckpt"%restore_from)

    for epoch in range(c):
        print("Epoch %d"%(epoch+1))
        for batch in feed(batches, epoch, batch_size, True):
            t = batch[1].astype('float')
            trainingStep.run(session=sess, feed_dict={X: batch[0], target: t})
            if( i % 100 == 0 ):
                [summ, lv] = sess.run([merged,loss], feed_dict={X: Xval, target: Yval})
                train_writer.add_summary(summ, i)
                saver.save(sess, "e:/data/tf_checkpoint/%s.ckpt"%clf_name)
            i += 1
            if i > 50000: break
        if i > 50000: break

        # saver.save(sess, "e:/data/tf_checkpoint/%s.ckpt"%clf_name)

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

def get_mesh(imshape, tile_size, stride):
    nr,nc = (imshape[0]-tile_size)//stride,(imshape[1]-tile_size)//stride
    yr,xr = np.arange(1,nr)*stride,np.arange(1,nc)*stride
    mesh = np.meshgrid(xr,yr)
    tiles = zip(mesh[0].flatten(), mesh[1].flatten())

    return tiles

def run_complete_test(saver, net, sess, clf_name, just_one=False):
    #basedir = 'e:/data/MITOS12/'
    #if False : basedir = os.path.join(basedir, 'Hamamatsu')
    
    for basedir in ['e:/data/MITOS12/', 'e:/data/MITOS12/Hamamatsu']:
        print(basedir)
        testdir = os.path.join(basedir,'test')
        clf = "e:/data/tf_checkpoint/%s.ckpt"%clf_name

        images = [f for f in os.listdir(testdir) if f.find('.bmp') >= 0]

        #imshape = (2084,2084)
        tile_size = 128
        stride = 64
        
        saver.restore(sess, clf)

        for imfile in images:
            im = plt.imread(os.path.join(testdir,imfile)).astype('float')/255.
            imshape = (im.shape[0],im.shape[1])
            tiles = get_mesh(imshape, tile_size, stride)
            pred = np.zeros(imshape) # channels : max(class0), max(class1)
            patch_X = np.zeros((1,128,128,3))
            result = np.zeros((imshape[0],imshape[1]))

            for t in tiles:
                patch_X[0] = im[t[1]:t[1]+tile_size, t[0]:t[0]+tile_size]
                
                Y = net.eval(session=sess, feed_dict={X: patch_X})
                Ysm = (np.exp(Y) / np.sum(np.exp(Y), axis=3)[:,:,:,np.newaxis])
                result[t[1]+stride//2:t[1]+tile_size-stride//2, t[0]+stride//2:t[0]+tile_size-stride//2] = Ysm[0,stride//2:tile_size-stride//2,stride//2:tile_size-stride//2,0]
                #break
            #break
            np.save(os.path.join(basedir, 'results/result_%s_%s.npy'%(clf_name,imfile)), result)
            plt.imsave(os.path.join(basedir, 'results/result_%s_%s.png'%(clf_name,imfile)), result)

import sys
import datetime
if __name__ == "__main__":
    # Get command line argument : train / test / detailed / fulltest

    if( len(sys.argv) > 1 ):
        action = sys.argv[1]
        batch_size = 1 if action == 'fulltest' else 20
        im_size = (128,128)
        lr = 1e-4
        eps = 0.1
        a = 1e-5
        kp = 0.5 if action == 'train' else 1.

        clf_name = "mitos12_resnet_detect6" if len(sys.argv) <= 2 else sys.argv[2]
        clf_from = None if len(sys.argv) <= 3 else sys.argv[3] 

        batches = ["E:\\data\\MITOS12\\train", "E:\\data\\MITOS12\\Hamamatsu\\train"]

        X = tf.placeholder(tf.float32, [batch_size,im_size[0],im_size[1],3])

        nets = get_network(X, kp)
        net = nets[0]

        saver = tf.train.Saver()

        target = tf.placeholder(tf.float32, [batch_size,2])
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=net))
        #loss = tf.losses.mean_squared_error(target, net)
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
        elif action == 'fulltest':
            run_complete_test(saver, net, sess, clf_name, True)
        else:
            print("Unknow command.")
            print("Usage : python resnet.py train|test|detailed|fulltest [clf_name] [clf_from]")

    else:
        print("Usage : python resnet.py train|test|detailed|fulltest [clf_name] [clf_from]")
