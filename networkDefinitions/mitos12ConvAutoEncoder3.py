import tensorflow as tf
from NetworkLayer import *
from WeightInit import WeightInit
from MITOS12Data import MITOS12Data
from matplotlib import pyplot as plt
import numpy as np
from sys import stdout
import os

from Network import Network

def C(pred,target):
    C = np.zeros((2,2))
    for i in range(len(pred)):
        C[np.array(pred[i]).argmax(),np.array(target[i]).argmax()] += 1
    return C

def mitos12ClassifierFromConvAutoEncoder3(train=False, resumeTraining=False, iterations=20000):
    autoEncoderModel = {
        'inputLayer' : ImageInputLayer(width=64,height=64,channels=3),
        'hiddenLayers' : [
            ConvolutionLayer(kernelsize=5, channels=3, features=12, stride=2, weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[64,64,3]),
            ConvolutionLayer(kernelsize=5, channels=12, features=48, stride=2,weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[32,32,12]),
            ConvolutionLayer(kernelsize=5, channels=48, features=192, stride=2,weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[16,16,48])
        ]
    }

    classifierModel = {
        'inputLayer': ImageInputLayer(width=8,height=8,channels=192),
        'hiddenLayers': [
            ImageToVectorLayer(imagesize=(8,8,192)),
            FullyConnectedLayer(inputsize=8*8*192,outputsize=200,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh),
            FullyConnectedLayer(inputsize=200,outputsize=100,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh)
            ],
        'outputLayer' : FullyConnectedLayer(inputsize=100,outputsize=2,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.softmax)
    }

    batch_size = 50
    autoEncoder = Network(autoEncoderModel, objective='reconstruction', batch_size=batch_size)

    clf = Network(classifierModel, objective='classification', batch_size=batch_size)
    clf.setupTraining("cross-entropy", "Adam")

    init = tf.initialize_all_variables()
    basedir = "/media/sf_E_DRIVE/Dropbox/ULB/Doctorat/ImageSet/MITOS12/"
    mitos12 = MITOS12Data(train_dirs=[os.path.join(basedir,d) for d in ["A00_v2", "A01_v2", "A02_v2", "A03_v2", "A04_v2"]])

    aesaver = tf.train.Saver(autoEncoder.getVariables())
    clfsaver = tf.train.Saver(clf.getVariables())
    sess = tf.Session()

    with sess.as_default():
        assert tf.get_default_session() is sess
        sess.run(init)
        aesaver.restore(sess, "/home/adrien/workspace/DeepNet/mitos12ConvAutoEncoder3.ckpt")

        if train==False or resumeTraining==True:
            clfsaver.restore(sess, "/home/adrien/workspace/DeepNet/mitos12ClfFromConvAutoEncoder3.ckpt")
        if train==True:
            for i in range(iterations):
                batch = mitos12.next_supervised_batch(batch_size)
                input_images = [b[0] for b in batch]
                
                batch_xs = autoEncoder.encode(input_images)
                batch_ys = [b[1] for b in batch]

                if i%1000==0:
                    cost = clf.cost.eval(feed_dict={clf.x: batch_xs, clf.target: batch_ys})
                    print "step %d, training cost %g"%(i,cost)
                    save_path = clfsaver.save(sess, "/home/adrien/workspace/DeepNet/mitos12ClfFromConvAutoEncoder3.ckpt")
            
                clf.train(batch_xs, batch_ys)

            save_path = clfsaver.save(sess, "/home/adrien/workspace/DeepNet/mitos12ClfFromConvAutoEncoder3.ckpt")

        # Eval :
        Cmat = np.zeros((2,2))
        for i in range(50):
            batch = mitos12.next_supervised_batch(batch_size)
            input_images = [b[0] for b in batch]
            
            batch_xs = autoEncoder.encode(input_images)
            batch_ys = [b[1] for b in batch]

            pred = clf.predict(batch_xs)
            Cmat += C(pred,batch_ys)

        print Cmat


def mitos12ConvAutoEncoder3(train=False, resumeTraining=False, iterations=20000):
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
    basedir = "/media/sf_E_DRIVE/Dropbox/ULB/Doctorat/ImageSet/MITOS12/"
    mitos12 = MITOS12Data(train_dirs=[os.path.join(basedir,d) for d in ["A00_v2", "A01_v2", "A02_v2", "A03_v2", "A04_v2"]])

    saver = tf.train.Saver()
    sess = tf.Session()

    with sess.as_default():
        assert tf.get_default_session() is sess
        sess.run(init)
        if train==False or resumeTraining==True:
            saver.restore(sess, "/home/adrien/workspace/DeepNet/mitos12ConvAutoEncoder3.ckpt")
        if train==True:
            for i in range(iterations):
                batch_xs = mitos12.next_batch(batch_size)
                batch_ys = batch_xs
                
                if i%1000 == 0:
                    cost = net.cost.eval(feed_dict={net.x: batch_xs, net.target: batch_ys})
                    print "step %d, training cost %g"%(i,cost)
                    save_path = saver.save(sess, "/home/adrien/workspace/DeepNet/mitos12ConvAutoEncoder3.ckpt")
            
                net.train(batch_xs, batch_ys)

            save_path = saver.save(sess, "/home/adrien/workspace/DeepNet/mitos12ConvAutoEncoder3.ckpt")

        xs = mitos12.next_batch(batch_size)

        es = net.encode(xs)
        
        for idx in range(len(es)):
            max_act = 0
            arg_max_act = 0
            es0 = es[idx]
            for t in np.arange(0,192):
                if( es0[:,:,t].sum() > max_act ):
                    max_act = es0[:,:,t].sum()
                    arg_max_act = t
            
            print arg_max_act
        return
        featt = np.zeros(es0.shape)
        featt[:,:,arg_max_act] = es0[:,:,arg_max_act]
        rs = net.decode([featt])
        print es0[:,:,arg_max_act]
        plt.figure()
        plt.gray()
        plt.imshow(es0[:,:,arg_max_act], interpolation='nearest')
        plt.figure()
        plt.imshow(rs[0]/rs[0].max(), interpolation='nearest')
        plt.show()
        return

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