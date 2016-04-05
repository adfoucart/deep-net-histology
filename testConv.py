import tensorflow as tf
from MITOS12Data import MITOS12Data
import numpy as np
from matplotlib import pyplot as plt
from Network import Network

from NetworkLayer import ConvolutionLayer, DeconvolutionLayer
from WeightInit import WeightInit

def normalize(x):
    return (x-x.min())/(x.max()-x.min())

mitos12 = MITOS12Data(train_dirs=["/media/sf_VirtualDropbox"])

batch_size = 50

x = tf.placeholder(tf.float32, [None, 64, 64, 3])
W_conv = tf.Variable(tf.truncated_normal([5,5,3,16], stddev=0.1))
b_conv = tf.Variable(tf.constant(0.1, shape=[16]))
h_conv = tf.nn.relu(tf.nn.conv2d(x, W_conv, strides=[1, 2, 2, 1], padding='SAME')+b_conv)

W_conv2 = tf.Variable(tf.truncated_normal([5,5,16,64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv, W_conv2, strides=[1, 2, 2, 1], padding='SAME')+b_conv2)

W_deconv2 = W_conv2
b_deconv2 = tf.constant(0., shape=[W_conv2.get_shape()[0].value])
h_deconv2 = tf.nn.relu(tf.nn.deconv2d(h_conv2, W_deconv2, [batch_size, 32, 32, 16], strides=[1, 2, 2, 1], padding="SAME"))

W_deconv = W_conv
b_deconv = tf.constant(0., shape=[W_conv.get_shape()[0].value])
h_deconv = tf.nn.relu(tf.nn.deconv2d(h_deconv2, W_deconv, [batch_size, 64, 64, 3], strides=[1, 2, 2, 1], padding="SAME"))

cost = tf.reduce_sum(tf.square(h_deconv-x))+tf.add_n([tf.nn.l2_loss(t) for t in [W_conv, W_conv2, b_conv, b_conv2]])
trainingStep = tf.train.AdamOptimizer(1e-4).minimize(cost)

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess, "/home/adrien/workspace/DeepNet/mitos12ConvAutoEncoder.ckpt")
    for e in range(200):
        if( e%100 == 0 ):
            batch_x = mitos12.next_batch(batch_size)
            print "Epoch %d, Cost : %f"%(e,cost.eval(feed_dict={x:batch_x}))

        batch_x = mitos12.next_batch(batch_size)
        trainingStep.run(feed_dict={x:batch_x})

    # save_path = saver.save(sess, "/home/adrien/workspace/DeepNet/mitos12ConvAutoEncoder.ckpt")

    # W_conv_ = W_conv.eval()
    # W_conv_ = (W_conv_ - W_conv_.min()) / (W_conv_.max()-W_conv_.min())

    # W_conv2_ = W_conv2.eval()
    # W_conv2_viz = np.zeros((5,5,3,64))

    # for w2 in range(64):
    #     for w1 in range(16):
    #         for c in range(3):
    #             W_conv2_viz[:,:,c,w2] += W_conv2_[:,:,w1,w2]*W_conv_[:,:,c,w1]

    # W_conv2_viz = normalize(W_conv2_viz)
    # plt.figure(0)
    # for i in range(64):
    #     plt.subplot(8,8,i+1)
    #     plt.imshow(W_conv2_viz[:,:,:,i])
    #     plt.axis('off')
    # plt.show()

    # plt.figure(1)
    # for i in xrange(16):
    #     plt.subplot(4,4,i+1)
    #     plt.imshow(W_conv_[:,:,:,i], interpolation="nearest")
    #     plt.axis('off')
    # plt.show()

    batch_x = mitos12.next_batch(batch_size)
    rx = h_deconv.eval(feed_dict={x:batch_x})
    plt.figure(1)
    for i in range(5):
        plt.subplot(5,2,2*i+1)
        plt.imshow(batch_x[i], interpolation='nearest')
        plt.axis('off')
        plt.subplot(5,2,2*i+2)
        plt.imshow(rx[i], interpolation='nearest')
        plt.axis('off')
    plt.show()
