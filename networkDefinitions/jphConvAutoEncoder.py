import tensorflow as tf
from NetworkLayer import *
from WeightInit import WeightInit
from JPHAlguesData import JPHAlguesData
from matplotlib import pyplot as plt
import numpy as np
from sys import stdout

from Network import Network

def jphConvAutoEncoder(train=False, resumeTraining=False):
    print "Initializing model"
    model = {
        'inputLayer' : ImageInputLayer(width=128, height=128, channels=3),
        'hiddenLayers': [
            ConvolutionLayer(kernelsize=5, channels=3, features=64, stride=4, weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[128,128,3]),
            ConvolutionLayer(kernelsize=5, channels=64, features=128, stride=2, weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[32,32,64]),
            ConvolutionLayer(kernelsize=5, channels=128, features=256, stride=2, weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[16,16,128])
        ]
    }

    batch_size = 50
    net = Network(model, objective='reconstruction', batch_size=batch_size)
    net.setupTraining("squared-diff", "Adam")

    init = tf.initialize_all_variables()
    jph = JPHAlguesData(train_dirs=["/home/adrien/workspace/JPHAlgues/tiles_10x"])

    saver = tf.train.Saver()
    sess = tf.Session()
    print "Starting session"

    with sess.as_default():
        assert tf.get_default_session() is sess
        sess.run(init)
        if train==False or resumeTraining==True:
            print "Restoring from file"
            saver.restore(sess, "/home/adrien/workspace/DeepNet/jphConvAutoEncoder.ckpt")
        if train==True:
            print "Starting training"
            for i in range(5000):
                batch_xs = jph.next_batch(batch_size)
                batch_ys = batch_xs
                
                if i%1000 == 0:
                    cost = net.cost.eval(feed_dict={net.x: batch_xs, net.target: batch_ys})
                    print "step %d, training cost %g"%(i,cost)
                    save_path = saver.save(sess, "/home/adrien/workspace/DeepNet/jphConvAutoEncoder.ckpt")
            
                net.train(batch_xs, batch_ys)

            print "Saving result"
            save_path = saver.save(sess, "/home/adrien/workspace/DeepNet/jphConvAutoEncoder.ckpt")

        print "Testing"
        xs = jph.next_batch(batch_size)
        rs = net.predict(xs)
        rs[0][rs[0]>1.] = 1.
        
        plt.figure(1)
        plt.subplot(2,3,1)
        plt.title('Original')
        plt.imshow(xs[0], interpolation='nearest')
        plt.axis('off')
        plt.subplot(2,3,2)
        plt.title('Reconstruction')
        plt.imshow(rs[0], interpolation='nearest')
        plt.axis('off')
        plt.gray()
        plt.subplot(2,3,4)
        plt.title('Diff - R')
        plt.imshow(np.abs(rs[0]-xs[0])[:,:,0], interpolation='nearest')
        plt.axis('off')
        plt.subplot(2,3,5)
        plt.title('Diff - G')
        plt.imshow(np.abs(rs[0]-xs[0])[:,:,1], interpolation='nearest')
        plt.axis('off')
        plt.subplot(2,3,6)
        plt.title('Diff - B')
        plt.imshow(np.abs(rs[0]-xs[0])[:,:,2], interpolation='nearest')
        plt.axis('off')
        plt.show()