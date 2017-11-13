##
# The goal of this version is to make a DETECTOR for slide defects.
# The final "segmented" image should be made by a sliding window technique.
##

from OpenSlideAnnotation import OpenSlideAnnotation
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.filters import gaussian

## -- Add Residual Layer -- ##
def add_residual(net_in, name, with_mp=True, width=128):
    neta = tf.contrib.layers.conv2d(net_in, width, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='%s/1'%name)
    netb = tf.contrib.layers.conv2d(neta, width, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='%s/2'%name)
    netc = tf.contrib.layers.conv2d(netb, net_in.get_shape()[3], 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='%s/3'%name)
    net = tf.add(net_in, netc, name='%s/add'%name)
    if with_mp:
        net = tf.contrib.layers.max_pool2d(net, 3, 2, 'SAME', scope='%s/mp'%name)
    return net

## -- Add Upsampling Layer -- ##
def add_up(net_in, name, width=None):
    if width==None:
        return tf.contrib.layers.conv2d_transpose(net_in, net_in.get_shape()[3], 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='%s'%name)
    else:
        return tf.contrib.layers.conv2d_transpose(net_in, width, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='%s'%name)

## -- Add Segmentation Layer -- ##
def add_seg(net_in, name, n_inter=64):
    net = tf.contrib.layers.conv2d(net_in, n_inter, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='%s/1'%name)
    net = tf.contrib.layers.conv2d(net, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='%s/2'%name)
    return net

## -- Data augmentation -- ##
def transform_batch(X,Y,isTraining,fuzzy=False):
    # X : input
    # Y : number of target pixels set as defect for each image
    p = np.minimum(Y/40.,1.) # prob = 1 if at least 40 pixels in the image are defect, 0 if 0 pixels, with linear progression in between
    Y2 = Y.copy()

    if not isTraining:
        return X,Y2

    params = np.random.random((X.shape[0], 3))
    params[:,0] = np.floor(params[:,0]*4)     # Horizontal & Vertical mirrors
    params[:,1] *= 0.2                        # Random noise max value (X'(i,j) = X(i,j) + random(i,j)*max_noise)
    params[:,2] = (params[:,2]-0.5)*0.2       # Illumination change (X'(i,j) = X(i,j) + illumination)

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

    # Fuzzy target
    if fuzzy:
        Y2 = gaussian(Y2,2)

    return X2,Y2

## -- Create the Network -- ##
def get_network(X):
    batch_size = X.get_shape()[0]
    im_size = tf.constant([X.get_shape().as_list()[1], X.get_shape().as_list()[2]])
    channels = X.get_shape()[3]

    ## slide_defect_detect_7
    # Widen the network
    net = tf.contrib.layers.conv2d(X, 128, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    res1 = add_residual(net, 'res1')
    res2 = add_residual(res1, 'res2', False)
    res3 = add_residual(res2, 'res3')

    up1 = add_up(res3, 'up1')

    res4 = add_residual(up1, 'res4', False)
    res5 = add_residual(res4, 'res5', False)
    
    up2 = add_up(res5, 'up2')

    res6 = add_residual(up2, 'res6', False)
    res7 = add_residual(res6, 'res7', False)

    # Segmentation layers
    seg = add_seg(res7, 'seg')

    return seg

## -- Data Feed -- ##
TS_SCAN3 = 1
TS_SCAN3_HE = 2

MAG_1_25 = 0
MAG_2_5 = 1
MAG_BOTH = 2

TARGET_SHARP = 0
TARGET_FUZZY = 1

BALANCE_NONE = 0
BALANCE_25 = 1
BALANCE_50 = 2

## -- Get a sample from the feed -- ## 
def get_sample(data, slide, batch_size, mag, target, balance, tile_size, isTraining):
    batch_X = np.zeros((batch_size,tile_size,tile_size,3))
    batch_Y = np.zeros((batch_size,tile_size,tile_size,1))

    # Draw mag levels
    levels = np.zeros((batch_size,))
    if( mag == MAG_1_25 ):
        levels[:] = 5
    elif( mag == MAG_2_5 ):
        levels[:] = 4
    else:
        levels[:] = np.round(np.random.random((batch_size,))+4)
    levels = levels.astype('int')

    # Prepare dataset balancing
    i = 0
    t = int(0.25*batch_size) if balance == BALANCE_25 else int(0.5*batch_size) if balance == BALANCE_50 else 0

    # Fill with compulsory artifact
    while i < t:
        level = levels[i]
        coords = data[slide][level]['coords']
        coord = coords[int(np.random.random()*coords.shape[0])]    # Draw random valid tile
        # Check if there is an artifact somewhere near the middle of the image (200-128 = 72 : the part of the tile that's in any translation variation)
        if( data[slide][level]['mask'][coord[0]+72:coord[1]-72,coord[2]+72:coord[3]-72].sum() > 0 ):
            rt = (np.random.random((2,))*np.array((72,72))).astype('int')
            batch_X[i,:,:,:] = data[slide][level]['im'][coord[0]+rt[0]:coord[0]+rt[0]+tile_size, coord[2]+rt[1]:coord[2]+rt[1]+tile_size,:3]
            batch_Y[i,:,:,0] = data[slide][level]['mask'][coord[0]+rt[0]:coord[0]+rt[0]+tile_size, coord[2]+rt[1]:coord[2]+rt[1]+tile_size]
            i += 1

    # Fill randomly the rest
    while i < batch_size:
        level = levels[i]
        coords = data[slide][level]['coords']
        coord = coords[int(np.random.random()*coords.shape[0])]
        rt = (np.random.random((2,))*np.array((72,72))).astype('int')
        batch_X[i,:,:,:] = data[slide][level]['im'][coord[0]+rt[0]:coord[0]+rt[0]+tile_size, coord[2]+rt[1]:coord[2]+rt[1]+tile_size,:3]
        batch_Y[i,:,:,0] = data[slide][level]['mask'][coord[0]+rt[0]:coord[0]+rt[0]+tile_size, coord[2]+rt[1]:coord[2]+rt[1]+tile_size]
        i += 1

    return transform_batch(batch_X/255,batch_Y, isTraining)

## -- Prepare dataset & yield samples -- ##
def feed(data, seed, batch_size, isTraining, ts, mag, tar, bal, tile_size):
    np.random.seed(seed)

    ndpis = list(data.keys())
    # Prepare random selection of slides
    N = len(ndpis)
    idxs = np.arange(N)
    np.random.shuffle(idxs)

    # For each slide : draw a random sample. Go through each slide once before starting again with new seed
    for idx in idxs:
        yield get_sample(data, ndpis[idx], batch_size, mag, tar, bal, tile_size, isTraining)

from datetime import datetime
import pickle
def train(saver, batch_dir, net, sess, clf_name, loss, merged, train_writer, restore_from, batch_size, hyper_params):
    # 1 mini-epoch = once through each slide = 20x18 = 360 examples.
    # For a million example : 2.777 mini-epochs 
    limit_mini_epochs = 3000 # -> 3000*18 = 54k mini-batches = 1.08 million training images

    ts = hyper_params['ts']
    mag = hyper_params['mag']
    tar = hyper_params['tar']
    bal = hyper_params['bal']
    tile_size = hyper_params['tile_size']

    # Preload all data
    print('Pre-loading data')
    data = {}
    with open(os.path.join(batch_dir, 'dataset.pkl'), 'rb') as f:
        data = pickle.load(f)
    ndpis = list(data.keys())
    # Filter for only Scan 3 if necessary
    if( ts == TS_SCAN3 ):
        ndpis = [ndpi for ndpi in ndpis if ndpi.find('Scan3') >= 0]
    print('%d slides in dataset.'%len(ndpis))

    print("Start training")

    Xval = np.load("E:/data/201702-scans/validation/Xval.npy")/255
    Yval = np.load("E:/data/201702-scans/validation/Yval_seg.npy")[:,:,:,np.newaxis]
    np.random.seed(0)
    idx = np.arange(Xval.shape[0])
    np.random.shuffle(idx)
    
    if restore_from != None:
        saver.restore(sess, "e:/data/tf_checkpoint/slide_defect_detect/%s.ckpt"%restore_from)

    best_val = 1e20
    i = 0
    for epoch in range(limit_mini_epochs):
        print("Epoch %d"%(epoch+1))
        for batch in feed(data, epoch, batch_size, True, ts, mag, tar, bal, tile_size):
            t = batch[1]
            trainingStep.run(session=sess, feed_dict={X: batch[0], target: t})
            if( i % 100 == 0 ):
                [summ, lv] = sess.run([merged,loss], feed_dict={X: Xval[idx[:batch_size]], target: Yval[idx[:batch_size]]})
                train_writer.add_summary(summ, i)
                saver.save(sess, "e:/data/tf_checkpoint/slide_defect_detect/%s.ckpt"%clf_name)
                if( lv < best_val ):
                    best_val = lv
                    saver.save(sess, "e:/data/tf_checkpoint/slide_defect_detect/%s_best.ckpt"%clf_name)
            i += 1

def test_val(saver, net, sess, clf_name, batch_size):
    ts = hyper_params['ts']
    mag = hyper_params['mag']
    tar = hyper_params['tar']
    bal = hyper_params['bal']
    tile_size = hyper_params['tile_size']

    print("Testing on validation data")

    Xval = np.load("E:/data/201702-scans/validation/Xval.npy")/255
    Yval = np.load("E:/data/201702-scans/validation/Yval_seg.npy")
    np.random.seed(0)
    #idx = np.arange(Xval.shape[0])
    #np.random.shuffle(idx)

    saver.restore(sess, "e:/data/tf_checkpoint/slide_defect_detect/%s.ckpt"%clf_name)
    
    Cmat = np.zeros((2,2))
    for i in range(8):
        Y = net.eval(session=sess, feed_dict={X: Xval[i*batch_size:(i+1)*batch_size]})
        plt.figure()
        for j in range(4):
            plt.subplot(4,3,j*3+1)
            plt.imshow(Xval[i*batch_size+j])
            plt.subplot(4,3,j*3+2)
            plt.imshow(Yval[i*batch_size+j,:,:])
            plt.subplot(4,3,j*3+3)
            plt.imshow(Y[j,:,:,0])
        plt.show()
        pred = Y[:,:,:,0] > 0.5
        #pred = Y.sum(axis=1).sum(axis=1) > 40 # 40 pixels = 1 -> 20 pixels = 0.5
        t = Yval[i*batch_size:(i+1)*batch_size,:,:]>0.5
        Cmat[0,0] += ((pred==0)*(t==0)).sum()
        Cmat[1,0] += ((pred==1)*(t==0)).sum()
        Cmat[0,1] += ((pred==0)*(t==1)).sum()
        Cmat[1,1] += ((pred==1)*(t==1)).sum()

    print("TN:", Cmat[0,0])
    print("FN:", Cmat[0,1])
    print("FP:", Cmat[1,0])
    print("TP:", Cmat[1,1])
    print("Accuracy:", (Cmat[0,0]+Cmat[1,1])/Cmat.sum())
    print("Precision:", Cmat[1,1]/(Cmat[1,1]+Cmat[1,0]))
    print("Recall:", Cmat[1,1]/(Cmat[1,1]+Cmat[0,1]))
    
    '''Y = net.eval(session=sess, feed_dict={X: Xval[idx[:batch_size]], target: Yval[idx[:batch_size]]})
    
    plt.figure()
    for i in range(batch_size):
        plt.subplot(4,5,i+1)
        plt.imshow(Xval[i,:,:,:])
        plt.title('%.2f / %.2f'%(Yval[i,0], Y[i,0]))
    plt.show()'''

import sys
if __name__ == "__main__":
    # Get command line argument : train / test
    if( len(sys.argv) > 1 ):
        action = sys.argv[1]
        batch_size = 20 if action == 'test' else 20
        tile_size = 128
        lr = 1e-3
        eps = 1e-8
        a = 1e-7

        # Hyper-Parameters
        ts = TS_SCAN3_HE
        mag = MAG_BOTH
        tar = TARGET_FUZZY
        bal = BALANCE_50
        tile_size = 128
        hyper_params = {'ts': ts, 'mag': mag, 'tar': tar, 'bal': bal, 'tile_size': tile_size}

        clf_name = "resnet_7" if len(sys.argv) <= 2 else sys.argv[2]
        clf_from = None if len(sys.argv) <= 3 else sys.argv[3] 

        train_dir = "E:\\data\\201702-scans\\train"

        X = tf.placeholder(tf.float32, [batch_size,tile_size,tile_size,3])

        net = get_network(X)
        saver = tf.train.Saver()

        target = tf.placeholder(tf.float32, [batch_size, tile_size, tile_size, 1])
        loss = tf.losses.mean_squared_error(target, net)
        loss = tf.add_n([loss] + [a*r for r in tf.losses.get_regularization_losses()])
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()

        optimizer = tf.train.AdamOptimizer(lr, epsilon=eps)
        trainingStep = optimizer.minimize(loss)

        sess = tf.Session()
        train_writer = tf.summary.FileWriter('e:/data/tf_summaries/slide_defect_seg/%s'%(clf_name), sess.graph)
        sess.run(tf.global_variables_initializer())

        tf.get_default_graph().finalize()
        
        if action == 'build':
            print("Built network.")
        elif action == 'train':
            # Save infos on run
            now = datetime.now().strftime("%Y%m%d%H%M")
            with open("e:/data/tf_checkpoint/slide_defect_detect/%s.%s.run.txt"%(now,clf_name), 'w') as runfile:
                runfile.write("%s\n"%now)
                runfile.write("Running classifier: %s\n"%clf_name)
                runfile.write("Training set : %d\n"%ts)
                runfile.write("Magnification : %d\n"%mag)
                runfile.write("Target : %d\n"%tar)
                runfile.write("Balance : %d\n"%bal)
                runfile.write("Tile Size : %d\n"%tile_size)
                runfile.write("Learning Rate : %f\n"%lr)
                runfile.write("Epsilon : %f\n"%eps)
                runfile.write("Alpha : %f\n"%a)
                if clf_from != None: runfile.write("clf_from: %s"%clf_from)
            train(saver, train_dir, net, sess, clf_name, loss, merged, train_writer, clf_from, batch_size, hyper_params)
        elif action == 'test':
            test_val(saver, net, sess, clf_name, batch_size)
        else:
            print("Unknow command.")
            print("Usage : python slide_defect_detect.py train|test [clf_name] [clf_from]")

    else:
        print("Usage : python slide_defect_detect.py train|test [clf_name] [clf_from]")
