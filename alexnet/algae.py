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

import tensorflow as tf
plt.switch_backend('agg')

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
    def __init__(self, batch_size, im_size, sess):
        net_data = np.load("bvlc_alexnet.npy").item()

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
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
        
        # Branch after conv4
        #maxpool4b
        conv41 = conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2)  # Removing ReLu
        maxpool4b = tf.nn.max_pool(conv41, ksize=[1, 6, 6, 1], strides=[1, 2, 2, 1], padding='VALID')
        fc4b = tf.reshape(maxpool4b, [batch_size, int(np.prod(maxpool4b.get_shape()[1:]))])

        self.output = fc8
        # self.output = fc4b

        self.sess = sess
        self.sess.run(tf.initialize_all_variables())

    def forward(self, X):
        return self.sess.run(self.output, feed_dict={self.x:X})


if __name__ == "__main__":
    # basedir = '/dropbox/ULB/Doctorat/workspace/DeepNet/alexnet'
    basedir = '.'
    algaedir = os.path.join(basedir,'../algae')
    file_dataset = [(f,f[:-8]) for f in os.listdir(algaedir) if f.find('_128.csv') > 0]
    train = file_dataset[:200]
    test = file_dataset[200:]

    file0 = train[10]
    im0 = plt.imread(os.path.join(algaedir,file0[1]))

    dataset = []
    # areas = np.zeros((im0.shape[0],im0.shape[1],4))
    # something = np.zeros((im0.shape[0],im0.shape[1]))
    with open(os.path.join(algaedir, file0[0]), 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            dataset += [(im0[int(row[0]):int(row[1]),int(row[2]):int(row[3]),:]-128, [float(row[4]),float(row[5]),float(row[6]),float(row[7])])]
            # areas[int(row[0]):int(row[1]),int(row[2]):int(row[3]),:] += np.asarray([float(row[4]),float(row[5]),float(row[6]),float(row[7])])
            # something[int(row[0]):int(row[1]),int(row[2]):int(row[3])] = 1.
    # areas[something==0.,3] = 1.

    '''plt.imsave("im0.png",im0)
    plt.imsave("areas.png", areas.argmax(axis=2))'''

    print "loading dataset for image..."
    X = np.array([d[0] for d in dataset])
    Y = [d[1] for d in dataset]

    '''ys = np.array(Y[:5000]).argmax(axis=1)
    Xindex = np.arange(5000)
    Xindex_sorted = np.zeros(Xindex.shape).astype('int')

    t = 0
    lims = [0]
    for i in range(4):
        r = ys==i
        Xindex_sorted[t:t+r.sum()] = Xindex[r]
        t += r.sum()
        lims += [t]

    for t in range(4):
        indexes = (np.random.random((9,))*(lims[t+1]-lims[t])).astype('int')
        Xs = X[Xindex_sorted[lims[t]:lims[t+1]]]
        plt.figure()
        for i,x in enumerate(indexes):
            plt.subplot(3,3,i+1)
            plt.imshow(Xs[x]+128)
        plt.savefig(os.path.join(basedir,'class_%d.png'%t))
    print lims'''

    print "Total size of dataset : %d"%len(dataset)
    batch_size = 100
    n_batches = 50

    print "Preparing AlexNet"
    sess = tf.Session()
    net = AlexNet(batch_size, X[0].shape, sess)
    n_features = net.output.get_shape()[-1]

    results = np.zeros((batch_size*n_batches,n_features))

    print "computing features for dataset"
    for i in range(n_batches):
        batch = X[batch_size*i:batch_size*(i+1)]
        if batch.shape[0] != batch_size: break
        results[batch_size*i:batch_size*(i+1),:] = net.forward(batch)

    sess.close()
    print results.shape, results.min(), results.max()

    # Ordering by class
    ys = np.array(Y[:batch_size*n_batches]).argmax(axis=1)
    results_sorted = np.zeros(results.shape)

    t = 0
    lims = []
    for i in range(4):
        r = results[ys==i]
        results_sorted[t:t+r.shape[0]] = r
        t += r.shape[0]
        lims += [t]

    plt.figure()
    plt.imshow(results_sorted.T)
    for l in lims:
        plt.plot([l, l], [0, results.shape[1]], 'k-', linewidth=2.0)
    plt.xlim([0, results.shape[0]])
    plt.ylim([0, results.shape[1]])
    # for i in range(0,4):
    #     plt.subplot(1,4,i+1)
    #     plt.imshow(results[ys==i].T)
    plt.savefig('features_sorted.png')


    '''train = file_dataset[:200]
    test = file_dataset[200:]
    print len(train),len(test),len(file_dataset)


    sess = tf.Session()
    net = AlexNet(1, (227,227), sess)
    features8 = net.forward([i])

    plt.figure()
    plt.plot(features8[0])
    plt.show()

    sess.close()'''