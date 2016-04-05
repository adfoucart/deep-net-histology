import tensorflow as tf
from NetworkLayer import *
from WeightInit import WeightInit
from MNISTData import MNISTData
from matplotlib import pyplot as plt
import numpy as np
from sys import stdout

from Network import Network

def mitos12ConvAutoEncoder3(train=False):
    model = {
        'inputLayer' : ImageInputLayer(width=64,height=64,channels=3),
        'hiddenLayers' : [
            ConvolutionLayer(kernelsize=5, channels=3, features=12, stride=2, weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[64,64,3]),
            ConvolutionLayer(kernelsize=5, channels=12, features=48, stride=2,weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[32,32,12]),
            ConvolutionLayer(kernelsize=5, channels=48, features=192, stride=2,weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[16,16,48])
        ]
    }

    batch_size = 50
    net = Network(model, objective='reconstruction', batch_size=batch_size)
    net.setupTraining("squared-diff", "Adam")

    init = tf.initialize_all_variables()
    mitos12 = MITOS12Data(train_dirs=["/media/sf_VirtualDropbox"])

    saver = tf.train.Saver()
    sess = tf.Session()

    with sess.as_default():
        assert tf.get_default_session() is sess
        sess.run(init)
        if train==False:
            saver.restore(sess, "/home/adrien/workspace/DeepNet/mitos12ConvAutoEncoder3.ckpt")
        else:
            for i in range(40000):
                batch_xs = mitos12.next_batch(batch_size)
                batch_ys = batch_xs
                
                if i%1000 == 0:
                    cost = net.cost.eval(feed_dict={net.x: batch_xs, net.target: batch_ys})
                    print "step %d, training cost %g"%(i,cost)
            
                net.train(batch_xs, batch_ys)

            save_path = saver.save(sess, "/home/adrien/workspace/DeepNet/mitos12ConvAutoEncoder3.ckpt")

        Ws = net.layers[1].W.eval()
        
        # xs = mitos12.next_batch(batch_size)
        # rs = net.predict(xs)
        # rs[0][rs[0]>1.] = 1.

        # es = net.getResponseAtLayer(xs, 1)
        plt.figure(1)
        plt.gray()
        for i in range(12):
            for c in range(3):
                plt.subplot(12,3,i*3+c+1)
                plt.imshow(Ws[:,:,c,i], interpolation='nearest')
                plt.axis('off')
        plt.show()
        
        # plt.figure(1)
        # plt.subplot(2,3,1)
        # plt.title('Original')
        # plt.imshow(xs[0], interpolation='nearest')
        # plt.axis('off')
        # plt.subplot(2,3,2)
        # plt.title('Reconstruction')
        # plt.imshow(rs[0], interpolation='nearest')
        # plt.axis('off')
        # plt.gray()
        # plt.subplot(2,3,4)
        # plt.title('Diff - R')
        # plt.imshow(np.abs(rs[0]-xs[0])[:,:,0], interpolation='nearest')
        # plt.axis('off')
        # plt.subplot(2,3,5)
        # plt.title('Diff - G')
        # plt.imshow(np.abs(rs[0]-xs[0])[:,:,1], interpolation='nearest')
        # plt.axis('off')
        # plt.subplot(2,3,6)
        # plt.title('Diff - B')
        # plt.imshow(np.abs(rs[0]-xs[0])[:,:,2], interpolation='nearest')
        # plt.axis('off')
        # plt.show()

mitos12ConvAutoEncoder3(False)