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
slide_defect_resnet & slide_defect_resnet_b
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

##########################

slide_defect_resnet_c
    # Widen the network
    net1 = tf.contrib.layers.conv2d(X, 256, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    # Residual unit 1
    net2a = tf.contrib.layers.conv2d(net1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_1')
    net2b = tf.contrib.layers.conv2d(net2a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_2')
    net2c = tf.contrib.layers.conv2d(net2b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_3')
    net2 = tf.add(net1, net2c)

    # Residual unit 2
    net3a = tf.contrib.layers.conv2d(net2, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_1')
    net3b = tf.contrib.layers.conv2d(net3a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_2')
    net3c = tf.contrib.layers.conv2d(net3b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_3')
    net3 = tf.add(net2, net3c)

    # Residual unit 3
    net4a = tf.contrib.layers.conv2d(net3, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_1')
    net4b = tf.contrib.layers.conv2d(net4a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_2')
    net4c = tf.contrib.layers.conv2d(net4b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_3')
    net4 = tf.add(net3, net4c)

    # Residual unit 4
    net5a = tf.contrib.layers.conv2d(net4, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_1')
    net5b = tf.contrib.layers.conv2d(net5a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_2')
    net5c = tf.contrib.layers.conv2d(net5b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_3')
    net5 = tf.add(net4, net5c)

    net6 = tf.contrib.layers.conv2d(net5, 64, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow1')
    net = tf.contrib.layers.conv2d(net6, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow')
    
    return [net]

##########################

slide_defect_resnet_d
    # Widen the network
    net1 = tf.contrib.layers.conv2d(X, 256, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    # Residual unit 1
    net2a = tf.contrib.layers.conv2d(net1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_1')
    net2b = tf.contrib.layers.conv2d(net2a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_2')
    net2c = tf.contrib.layers.conv2d(net2b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_3')
    net2 = tf.add(net1, net2c)
    net2mp = tf.contrib.layers.max_pool2d(net2, 3, 1, 'SAME', scope='net2/mp')

    # Residual unit 2
    net3a = tf.contrib.layers.conv2d(net2mp, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_1')
    net3b = tf.contrib.layers.conv2d(net3a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_2')
    net3c = tf.contrib.layers.conv2d(net3b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_3')
    net3 = tf.add(net2mp, net3c)
    net3mp = tf.contrib.layers.max_pool2d(net3, 3, 1, 'SAME', scope='net3/mp')

    # Residual unit 3
    net4a = tf.contrib.layers.conv2d(net3mp, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_1')
    net4b = tf.contrib.layers.conv2d(net4a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_2')
    net4c = tf.contrib.layers.conv2d(net4b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_3')
    net4 = tf.add(net3mp, net4c)

    # Residual unit 4
    net5a = tf.contrib.layers.conv2d(net4, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_1')
    net5b = tf.contrib.layers.conv2d(net5a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_2')
    net5c = tf.contrib.layers.conv2d(net5b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_3')
    net5 = tf.add(net4, net5c)

    net6 = tf.contrib.layers.conv2d(net5, 64, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow1')
    net = tf.contrib.layers.conv2d(net6, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow')
    
    return [net]

##########################

slide_defect_alexnet_fc
    # AlexNet -> conv 4
    conv1 = tf.contrib.layers.conv2d(X, 96, 11, 4, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='alexnet/conv1')
    lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, bias=1., alpha=2e-05, beta=0.75, name="alexnet/lrn1")
    mp1 = tf.contrib.layers.max_pool2d(lrn1, 3, 2, 'SAME', scope='alexnet/mp1')
    conv2 = tf.contrib.layers.conv2d(mp1, 256, 5, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='alexnet/conv2')
    lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=2, bias=1., alpha=2e-05, beta=0.75, name="alexnet/lrn2")
    mp2 = tf.contrib.layers.max_pool2d(lrn2, 3, 2, 'SAME', scope='alexnet/mp2')
    conv3 = tf.contrib.layers.conv2d(mp2, 384, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='alexnet/conv3')
    conv4 = tf.contrib.layers.conv2d(conv3, 384, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='alexnet/conv4')
    
    # Upsampling
    up5 = tf.contrib.layers.conv2d_transpose(conv4, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up5')
    up6 = tf.contrib.layers.conv2d_transpose(up5, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up6')
    up7 = tf.contrib.layers.conv2d_transpose(up6, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up7')
    up8 = tf.contrib.layers.conv2d_transpose(up7, 96, 3, 4, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up8')
    
    # Narrow
    narrow9 = tf.contrib.layers.conv2d(up8, 64, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow9')
    net = tf.contrib.layers.conv2d(narrow9, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow')
    
    return [net]

##########################

slide_defect_alexnet_fcres
    # AlexNet Residual -> conv 4
    conv1_1 = tf.contrib.layers.conv2d(X, 96, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='alexnet/conv1_1')
    conv1_2 = tf.contrib.layers.conv2d(conv1_1, 96, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='alexnet/conv1_2')
    conv1_3 = tf.contrib.layers.conv2d(conv1_2, 96, 3, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='alexnet/conv1_3')
    conv1_4 = tf.contrib.layers.conv2d(conv1_3, 96, 3, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='alexnet/conv1_4')
    conv1_res = tf.contrib.layers.conv2d(X, 96, 1, 4, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='alexnet/conv1_res')
    conv1 =  tf.add(conv1_4, conv1_res, name="alexnet/add1")
    mp1 = tf.contrib.layers.max_pool2d(conv1, 3, 2, 'SAME', scope='alexnet/mp1')
    
    conv2_1 = tf.contrib.layers.conv2d(mp1, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='alexnet/conv2_1')
    conv2_2 = tf.contrib.layers.conv2d(conv2_1, 256, 3, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='alexnet/conv2_2')
    conv2_res = tf.contrib.layers.conv2d(mp1, 256, 1, 2, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='alexnet/conv2_res')
    conv2 =  tf.add(conv2_2, conv2_res, name='alexnet/add2')
    mp2 = tf.contrib.layers.max_pool2d(conv2, 3, 2, 'SAME', scope='alexnet/mp2')

    conv3 = tf.contrib.layers.conv2d(mp2, 384, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='alexnet/conv3')
    conv4 = tf.contrib.layers.conv2d(conv3, 384, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='alexnet/conv4')
    
    # Upsampling
    up5 = tf.contrib.layers.conv2d_transpose(conv4, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up5')
    up6 = tf.contrib.layers.conv2d_transpose(up5, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up6')
    up7 = tf.contrib.layers.conv2d_transpose(up6, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up7')
    up8 = tf.contrib.layers.conv2d_transpose(up7, 96, 3, 4, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up8')
    
    # Narrow
    narrow9 = tf.contrib.layers.conv2d(up8, 64, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow9')
    net = tf.contrib.layers.conv2d(narrow9, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow')
    
    return [net]

##########################

slide_defect_resnet_stride
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
    net4a = tf.contrib.layers.conv2d(net3mp, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/1')
    net4b = tf.contrib.layers.conv2d(net4a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/2')
    net4c = tf.contrib.layers.conv2d(net4b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/3')
    net4 = tf.add(net3mp, net4c, name='res3/add')

    up1 = tf.contrib.layers.conv2d_transpose(net4, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up1')
    
    # Residual unit 4
    net5a = tf.contrib.layers.conv2d(up1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/1')
    net5b = tf.contrib.layers.conv2d(net5a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/2')
    net5c = tf.contrib.layers.conv2d(net5b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/3')
    net5 = tf.add(up1, net5c, name='res4/add')

    up2 = tf.contrib.layers.conv2d_transpose(net5, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up2')

    net6 = tf.contrib.layers.conv2d(up2, 64, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow1')
    net = tf.contrib.layers.conv2d(net6, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow')
    
    return [net]

##########################

slide_defect_resnet_stride_dropout
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

    # Dropout
    drop1 = tf.nn.dropout(net3mp, keep_prob=kp, name='drop1')

    # Residual unit 3
    net4a = tf.contrib.layers.conv2d(drop1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/1')
    net4b = tf.contrib.layers.conv2d(net4a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/2')
    net4c = tf.contrib.layers.conv2d(net4b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/3')
    net4 = tf.add(drop1, net4c, name='res3/add')

    up1 = tf.contrib.layers.conv2d_transpose(net4, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up1')
    
    # Residual unit 4
    net5a = tf.contrib.layers.conv2d(up1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/1')
    net5b = tf.contrib.layers.conv2d(net5a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/2')
    net5c = tf.contrib.layers.conv2d(net5b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/3')
    net5 = tf.add(up1, net5c, name='res4/add')

    # Dropout
    drop2 = tf.nn.dropout(net5, keep_prob=kp, name='drop2')

    up2 = tf.contrib.layers.conv2d_transpose(drop2, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up2')

    net6 = tf.contrib.layers.conv2d(up2, 64, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow1')
    net = tf.contrib.layers.conv2d(net6, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow')
    
    return [net]

##########################

slide_defect_resnet_7
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
    net4a = tf.contrib.layers.conv2d(net3mp, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/1')
    net4b = tf.contrib.layers.conv2d(net4a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/2')
    net4c = tf.contrib.layers.conv2d(net4b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/3')
    net4 = tf.add(net3mp, net4c, name='res3/add')
    net4mp = tf.contrib.layers.max_pool2d(net4, 3, 2, 'SAME', scope='res3/mp')
    
    up1 = tf.contrib.layers.conv2d_transpose(net4mp, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up1')
    
    # Residual unit 4
    net5a = tf.contrib.layers.conv2d(up1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/1')
    net5b = tf.contrib.layers.conv2d(net5a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/2')
    net5c = tf.contrib.layers.conv2d(net5b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/3')
    net5 = tf.add(up1, net5c, name='res4/add')
    
    up2 = tf.contrib.layers.conv2d_transpose(net5, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up2')
    
    # Residual unit 5
    net6a = tf.contrib.layers.conv2d(up2, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res5/1')
    net6b = tf.contrib.layers.conv2d(net6a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res5/2')
    net6c = tf.contrib.layers.conv2d(net6b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res5/3')
    net6 = tf.add(up2, net6c, name='res5/add')
    
    up3 = tf.contrib.layers.conv2d_transpose(net6, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up3')
    
    net7 = tf.contrib.layers.conv2d(up3, 64, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='seg1')
    net = tf.contrib.layers.conv2d(net7, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='seg')
    
    return [net]

##########################

slide_defect_resnet_10
slide_defect_resnet_10_2
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
    net4a = tf.contrib.layers.conv2d(net3mp, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/1')
    net4b = tf.contrib.layers.conv2d(net4a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/2')
    net4c = tf.contrib.layers.conv2d(net4b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/3')
    net4 = tf.add(net3mp, net4c, name='res3/add')
    net4mp = tf.contrib.layers.max_pool2d(net4, 3, 2, 'SAME', scope='res3/mp')
    
    up1 = tf.contrib.layers.conv2d_transpose(net4mp, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up1')
    
    # Residual unit 4
    net5a = tf.contrib.layers.conv2d(up1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/1')
    net5b = tf.contrib.layers.conv2d(net5a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/2')
    net5c = tf.contrib.layers.conv2d(net5b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/3')
    net5 = tf.add(up1, net5c, name='res4/add')

    # Residual unit 5
    net6a = tf.contrib.layers.conv2d(net5, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res5/1')
    net6b = tf.contrib.layers.conv2d(net6a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res5/2')
    net6c = tf.contrib.layers.conv2d(net6b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res5/3')
    net6 = tf.add(net5, net6c, name='res5/add')

    # Residual unit 6
    net7a = tf.contrib.layers.conv2d(net6, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res6/1')
    net7b = tf.contrib.layers.conv2d(net7a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res6/2')
    net7c = tf.contrib.layers.conv2d(net7b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res6/3')
    net7 = tf.add(net6, net7c, name='res6/add')

    # Residual unit 7
    net8a = tf.contrib.layers.conv2d(net7, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res7/1')
    net8b = tf.contrib.layers.conv2d(net8a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res7/2')
    net8c = tf.contrib.layers.conv2d(net8b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res7/3')
    net8 = tf.add(net7, net8c, name='res7/add')

    up2 = tf.contrib.layers.conv2d_transpose(net8, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up2')
    
    # Residual unit 8
    net9a = tf.contrib.layers.conv2d(up2, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res8/1')
    net9b = tf.contrib.layers.conv2d(net9a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res8/2')
    net9c = tf.contrib.layers.conv2d(net9b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res8/3')
    net9 = tf.add(up2, net6c, name='res8/add')
    
    up3 = tf.contrib.layers.conv2d_transpose(net9, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up3')
    
    net10 = tf.contrib.layers.conv2d(up3, 64, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='seg/1')
    net = tf.contrib.layers.conv2d(net10, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='seg/2')
    
    return [net]

##########################

slide_defect_resnet_wide
    # Widen the network
    net1 = tf.contrib.layers.conv2d(X, 256, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    # RESIDUAL 1

    # -- A
    res1a1 = tf.contrib.layers.conv2d(net1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/a/1')
    res1a2 = tf.contrib.layers.conv2d(res1a1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/a/2')
    res1a3 = tf.contrib.layers.conv2d(res1a2, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/a/3')
    res1aadd = tf.add(net1, res1a3, name='res1/a/add')
    res1amp = tf.contrib.layers.max_pool2d(res1aadd, 3, 2, 'SAME', scope='res1/a/mp')
    
    # -- B
    res1b1 = tf.contrib.layers.conv2d(net1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/b/1')
    res1b2 = tf.contrib.layers.conv2d(res1b1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/b/2')
    res1b3 = tf.contrib.layers.conv2d(res1b2, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/b/3')
    res1badd = tf.add(net1, res1b3, name='res1/b/add')
    res1bmp = tf.contrib.layers.max_pool2d(res1badd, 3, 2, 'SAME', scope='res1/b/mp')
    
    # -- C
    res1c1 = tf.contrib.layers.conv2d(net1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/c/1')
    res1c2 = tf.contrib.layers.conv2d(res1c1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/c/2')
    res1c3 = tf.contrib.layers.conv2d(res1c2, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/c/3')
    res1cadd = tf.add(net1, res1c3, name='res1/c/add')
    res1cmp = tf.contrib.layers.max_pool2d(res1cadd, 3, 2, 'SAME', scope='res1/c/mp')

    # -- Concatenation
    res1concat = tf.concat([res1amp, res1bmp, res1cmp], 0, name='res1/concat')

    # RESIDUAL 2

    # -- A
    res2a1 = tf.contrib.layers.conv2d(res1concat, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/a/1')
    res2a2 = tf.contrib.layers.conv2d(res2a1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/a/2')
    res2a3 = tf.contrib.layers.conv2d(res2a2, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/a/3')
    res2aadd = tf.add(res1concat, res2a3, name='res2/a/add')
    res2amp = tf.contrib.layers.max_pool2d(res2aadd, 3, 2, 'SAME', scope='res2/a/mp')

    # -- B
    res2b1 = tf.contrib.layers.conv2d(res1concat, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/b/1')
    res2b2 = tf.contrib.layers.conv2d(res2b1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/b/2')
    res2b3 = tf.contrib.layers.conv2d(res2b2, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/b/3')
    res2badd = tf.add(res1concat, res2b3, name='res2/b/add')
    res2bmp = tf.contrib.layers.max_pool2d(res2badd, 3, 2, 'SAME', scope='res2/b/mp')

    # -- C
    res2c1 = tf.contrib.layers.conv2d(res1concat, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/c/1')
    res2c2 = tf.contrib.layers.conv2d(res2c1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/c/2')
    res2c3 = tf.contrib.layers.conv2d(res2c2, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/c/3')
    res2cadd = tf.add(res1concat, res2c3, name='res2/c/add')
    res2cmp = tf.contrib.layers.max_pool2d(res2cadd, 3, 2, 'SAME', scope='res2/c/mp')

    # Concatenation
    res2concat = tf.concat([res2amp, res2bmp, res2cmp], 0, name='res2/concat')

    # Upsampling
    up1 = tf.contrib.layers.conv2d_transpose(res2concat, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up1')
        
    # RESIDUAL 3
    res3a = tf.contrib.layers.conv2d(up1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/1')
    res3b = tf.contrib.layers.conv2d(res3a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/2')
    res3c = tf.contrib.layers.conv2d(res3b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/3')
    res3add = tf.add(up1, res3c, name='res3/add')

    # Upsampling
    up2 = tf.contrib.layers.conv2d_transpose(net8, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up2')
    
    # Segmentation    
    seg1 = tf.contrib.layers.conv2d(up2, 64, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='seg/1')
    seg2 = tf.contrib.layers.conv2d(seg1, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='seg/2')
    
    return [seg2]

##########################

slide_defect_resnet_wide_short
    # Widen the network
    net1 = tf.contrib.layers.conv2d(X, 512, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    # Residual unit 1
    net2a = tf.contrib.layers.conv2d(net1, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/1')
    net2b = tf.contrib.layers.conv2d(net2a, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/2')
    net2c = tf.contrib.layers.conv2d(net2b, 512, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/3')
    net2 = tf.add(net1, net2c, name='res1/add')
    net2mp = tf.contrib.layers.max_pool2d(net2, 3, 2, 'SAME', scope='res1/mp')
    
    # Residual unit 2
    net3a = tf.contrib.layers.conv2d(net2mp, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/1')
    net3b = tf.contrib.layers.conv2d(net3a, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/2')
    net3c = tf.contrib.layers.conv2d(net3b, 512, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/3')
    net3 = tf.add(net2mp, net3c, name='res2/add')
    net3mp = tf.contrib.layers.max_pool2d(net3, 3, 2, 'SAME', scope='res2/mp')
        
    up1 = tf.contrib.layers.conv2d_transpose(net3mp, 512, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up1')
    
    # Residual unit 4
    net5a = tf.contrib.layers.conv2d(up1, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/1')
    net5b = tf.contrib.layers.conv2d(net5a, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/2')
    net5c = tf.contrib.layers.conv2d(net5b, 512, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/3')
    net5 = tf.add(up1, net5c, name='res4/add')
    
    up2 = tf.contrib.layers.conv2d_transpose(net5, 512, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up2')
    
    # Residual unit 5
    net6a = tf.contrib.layers.conv2d(up2, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res5/1')
    net6b = tf.contrib.layers.conv2d(net6a, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res5/2')
    net6c = tf.contrib.layers.conv2d(net6b, 512, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res5/3')
    net6 = tf.add(up2, net6c, name='res5/add')
    
    net7 = tf.contrib.layers.conv2d(net6, 64, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='seg1')
    net = tf.contrib.layers.conv2d(net7, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='seg')
    
    return [net]

##########################

slide_defect_resnet_13
    # Widen the network
    net = tf.contrib.layers.conv2d(X, 128, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    res1 = add_residual(net, 'res1')
    res2 = add_residual(res1, 'res2')
    res3 = add_residual(res2, 'res3')
    res4 = add_residual(res3, 'res4', False)

    up1 = add_up(res4, 'up1')

    res5 = add_residual(up1, 'res5', False)
    res6 = add_residual(res5, 'res6', False)
    res7 = add_residual(res6, 'res7', False)
    
    up2 = add_up(res7, 'up2')

    res8 = add_residual(up2, 'res8', False)
    res9 = add_residual(res8, 'res9', False)
    res10 = add_residual(res9, 'res10', False)

    up3 = add_up(res10, 'up3')

    res11 = add_residual(up3, 'res11', False)
    res12 = add_residual(res11, 'res12', False)
    res13 = add_residual(res12, 'res13', False)

    net = add_seg(res13, 'seg')

##########################

slide_defect_resnet_13b
    # Widen the network
    net = tf.contrib.layers.conv2d(X, 256, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    res1 = add_residual(net, 'res1')
    res2 = add_residual(res1, 'res2')
    res3 = add_residual(res2, 'res3')
    res4 = add_residual(res3, 'res4', False)

    up1 = add_up(res4, 'up1', 128)

    res5 = add_residual(up1, 'res5', False)
    res6 = add_residual(res5, 'res6', False)
    res7 = add_residual(res6, 'res7', False)
    
    up2 = add_up(res7, 'up2', 128)

    res8 = add_residual(up2, 'res8', False)
    res9 = add_residual(res8, 'res9', False)
    res10 = add_residual(res9, 'res10', False)

    up3 = add_up(res10, 'up3', 128)

    res11 = add_residual(up3, 'res11', False)
    res12 = add_residual(res11, 'res12', False)
    res13 = add_residual(res12, 'res13', False)

    net = add_seg(res13, 'seg')
    
    return [net]

##########################

slide_defect_resnet_branch
    # Widen the network
    net = tf.contrib.layers.conv2d(X, 128, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    resa1 = add_residual(net, 'resa1', True, 64)
    resa2 = add_residual(resa1, 'resa2', False, 64)
    resa3 = add_residual(resa2, 'resa3', False, 64)
    upa = add_up(resa3, 'upa')

    resb1 = add_residual(net, 'resb1', True, 64)
    resb2 = add_residual(resb1, 'resb2', True, 128)
    resb3 = add_residual(resb2, 'resb3', False, 128)
    upb1 = add_up(resb3, 'upb1')
    upb2 = add_up(upb1, 'upb2')

    resc1 = add_residual(net, 'resc1', True, 64)
    resc2 = add_residual(resc1, 'resc2', True, 128)
    resc3 = add_residual(resc2, 'resc3', True, 256)
    upc1 = add_up(resc3, 'upc1')
    upc2 = add_up(upc1, 'upc2')
    upc3 = add_up(upc2, 'upc3')

    net = tf.add(upa, upb2, name='add_abc/1')
    net = tf.add(net, upc3, name='add_abc/2')

    res4 = add_residual(net, 'res4', False)
    res5 = add_residual(res4, 'res5', False)
    res6 = add_residual(res5, 'res6', False)

    net = add_seg(res6, 'seg', 64)

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
    net = tf.contrib.layers.conv2d(X, 128, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    resa1 = add_residual(net, 'resa1', True, 64)
    resa2 = add_residual(resa1, 'resa2', False, 64)
    resa3 = add_residual(resa2, 'resa3', False, 64)
    upa = add_up(resa3, 'upa')

    resb1 = add_residual(net, 'resb1', True, 64)
    resb2 = add_residual(resb1, 'resb2', True, 128)
    resb3 = add_residual(resb2, 'resb3', False, 128)
    upb1 = add_up(resb3, 'upb1')
    upb2 = add_up(upb1, 'upb2')

    resc1 = add_residual(net, 'resc1', True, 64)
    resc2 = add_residual(resc1, 'resc2', True, 128)
    resc3 = add_residual(resc2, 'resc3', True, 256)
    upc1 = add_up(resc3, 'upc1')
    upc2 = add_up(upc1, 'upc2')
    upc3 = add_up(upc2, 'upc3')

    net = tf.add(upa, upb2, name='add_abc/1')
    net = tf.add(net, upc3, name='add_abc/2')

    res4 = add_residual(net, 'res4', False)
    res5 = add_residual(res4, 'res5', False)
    res6 = add_residual(res5, 'res6', False)

    net = add_seg(res6, 'seg', 64)

    return [net]

def transform_batch(X,Y,isTraining):
    if not isTraining:
        return X,Y
    params = np.random.random((X.shape[0], 3))
    params[:,0] = np.floor(params[:,0]*4)
    params[:,1] *= 0.2
    params[:,2] = (params[:,2]-0.5)*0.2

    X2 = X.copy()
    Y2 = Y.copy()

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

    return X2,Y2

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
    batch_size = min(batch_size, 20)

    epoch = 40
    for batch in feed(batches, epoch, batch_size, False):
        t = batch[1].astype('float')
        # t[t==0] = -1.
        # t[t>0] = 1.
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
    batch_size = min(batch_size, 20)

    print("Start training")

    if restore_from != None:
        saver.restore(sess, "e:/data/tf_checkpoint/%s.ckpt"%restore_from)

    for epoch in range(c):
        print("Epoch %d"%(epoch+1))
        for batch in feed(batches, epoch, batch_size, True):
            t = batch[1].astype('float')
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
        lr = 1e-2
        eps = 0.1
        a = 0#1e-5
        kp = 0.5 if action == 'train' else 1.

        clf_name = "slide_defect_resnet" if len(sys.argv) <= 2 else sys.argv[2]
        clf_from = None if len(sys.argv) <= 3 else sys.argv[3] 

        batches = "E:\\data\\201702-scans\\images"

        X = tf.placeholder(tf.float32, [batch_size,im_size[0],im_size[1],3])

        #net,net6,net5,net5c,net4,net4c,net3,net3c,net2,net2c,net1 = get_network(X)
        nets = get_network(X, kp)
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
            detailed_test(saver, sess, net, clf_name, batches)
        elif action == 'fulltest':
            run_complete_test(saver, net, sess, clf_name, True)
        else:
            print("Unknow command.")
            print("Usage : python resnet.py train|test|detailed|fulltest [clf_name] [clf_from]")

    else:
        print("Usage : python resnet.py train|test|detailed|fulltest [clf_name] [clf_from]")
