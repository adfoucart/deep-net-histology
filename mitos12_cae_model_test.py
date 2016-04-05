import tensorflow as tf
from MITOS12Data import MITOS12Data
from matplotlib import pyplot as plt
import numpy as np
from sys import stdout
from NetworkLayer import *
from WeightInit import WeightInit
from Network import Network
from AutoEncoderNetwork import Mitos12ConvAutoEncoder3NetworkDefinition as Mitos12_CAE_Model

def mitos12_cae_model_test():
    model = {
        'inputLayer' : ImageInputLayer(width=64,height=64,channels=3),
        'hiddenLayers' : [
            ConvolutionLayer(kernelsize=5, channels=3, features=12, stride=2, weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[64,64,3]),
            ConvolutionLayer(kernelsize=5, channels=12, features=48, stride=2,weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[32,32,12]),
            ConvolutionLayer(kernelsize=5, channels=48, features=192, stride=2,weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[16,16,48])
        ]
    }

    batch_size = 50
    net = Network(Mitos12_CAE_Model, objective='reconstruction', batch_size=batch_size)
    net.setupTraining("squared-diff", "Adam")

    init = tf.initialize_all_variables()
    mitos12 = MITOS12Data(train_dirs=["/media/sf_VirtualDropbox"])

    saver = tf.train.Saver()
    sess = tf.Session()

    with sess.as_default():
        assert tf.get_default_session() is sess
        sess.run(init)
        saver.restore(sess, "/home/adrien/workspace/DeepNet/mitos12ConvAutoEncoder3.ckpt")

        xs = mitos12.next_batch(batch_size)
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

mitos12_cae_model_test()