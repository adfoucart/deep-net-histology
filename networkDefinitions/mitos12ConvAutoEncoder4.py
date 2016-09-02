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

def mitos12ClassifierFromConvAutoEncoder4(train=False, resumeTraining=False, iterations=20000):
    autoEncoderModel = {
        'inputLayer' : ImageInputLayer(width=128,height=128,channels=3),
        'hiddenLayers' : [
            ConvolutionLayer(kernelsize=15, channels=3, features=12, stride=4, weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[128,128,3]),
            ConvolutionLayer(kernelsize=7, channels=12, features=40, stride=2,weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[32,32,12]),
            ConvolutionLayer(kernelsize=5, channels=40, features=80, stride=2,weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[16,16,40])
        ]
    }

    classifierModel = {
        'inputLayer': ImageInputLayer(width=8,height=8,channels=80),
        'hiddenLayers': [
            ImageToVectorLayer(imagesize=(8,8,80)),
            FullyConnectedLayer(inputsize=8*8*80,outputsize=100,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh),
            # FullyConnectedLayer(inputsize=100,outputsize=100,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh)
            ],
        'outputLayer' : FullyConnectedLayer(inputsize=100,outputsize=2,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.softmax)
    }

    batch_size = 500
    autoEncoder = Network(autoEncoderModel, objective='reconstruction', batch_size=batch_size)
    autoEncoderName = "mitos12ConvAutoEncoder4WithDistortion"

    clf = Network(classifierModel, objective='classification', batch_size=batch_size)
    clf.setupTraining("cross-entropy", "Adam", a=0.999, lr=1e-4)
    clfName = "mitos12ClfFromConvAutoEncoder4WithDistortion"

    init = tf.initialize_all_variables()
    basedir = "/media/sf_E_DRIVE/Dropbox/ULB/Doctorat/ImageSet/MITOS12/"
    mitos12 = MITOS12Data(train_dirs=[os.path.join(basedir,d) for d in ["A00_v2", "A01_v2", "A02_v2", "A03_v2", "A04_v2"]], chunksize=(128,128))

    aesaver = tf.train.Saver(autoEncoder.getVariables())
    clfsaver = tf.train.Saver(clf.getVariables())
    sess = tf.Session()

    with sess.as_default():
        assert tf.get_default_session() is sess
        sess.run(init)
        aesaver.restore(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%autoEncoderName)

        if train==False or resumeTraining==True:
            clfsaver.restore(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clfName)
        if train==True:
            for i in range(iterations):
                batch = mitos12.next_supervised_batch(batch_size, pMitosis=0.5)
                input_images = [b[0] for b in batch]
                
                batch_xs = autoEncoder.encode(input_images)
                batch_ys = [b[1] for b in batch]

                if i%1000==0:
                    cost = clf.cost.eval(feed_dict={clf.x: batch_xs, clf.target: batch_ys})
                    loss = clf.loss.eval(feed_dict={clf.x: batch_xs, clf.target: batch_ys})
                    l2loss = clf.l2loss.eval()
                    print "step %d, training cost %g, loss %g, l2loss %g"%(i,cost,loss,l2loss)
                    with open("/home/adrien/workspace/DeepNet/%s_results.txt"%clfName, "a") as resFile:
                        resFile.write("step %d, training cost %g, loss %g, l2loss %g\n"%(i,cost,loss,l2loss))
                    save_path = clfsaver.save(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clfName)
            
                clf.train(batch_xs, batch_ys)

            save_path = clfsaver.save(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clfName)

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

        im_, path, basename = mitos12.images[18]
        im = np.array(im_)
        stride = 15
        rangex = np.arange(0,im.shape[0]-128,stride)
        rangey = np.arange(0,im.shape[1]-128,stride)
        ts = [(t/len(rangey), t%len(rangey)) for t in range(len(rangex)*len(rangey))]
        chunks = [im[tx*stride:tx*stride+128,ty*stride:ty*stride+128,:] for tx,ty in ts]
        chunksPos = [(tx*stride,ty*stride) for tx,ty in ts]
        pMitosis = np.zeros((im.shape[0], im.shape[1], 3))

        print len(chunks)
        for t in range(len(chunks)/50):
            batch = chunks[t*50:t*50+50]
            batch_xs = autoEncoder.encode(batch)
            is_mitosis = clf.predict(batch_xs)
            for i,p in enumerate(is_mitosis):
                cp = chunksPos[t*50+i]
                pMitosis[cp[0]:cp[0]+128, cp[1]:cp[1]+128, 0] += p[0]
                pMitosis[cp[0]:cp[0]+128, cp[1]:cp[1]+128, 1] += p[1]
                pMitosis[cp[0]:cp[0]+128, cp[1]:cp[1]+128, 2] += 1

        plt.figure()
        plt.gray()
        plt.imshow(pMitosis[:,:,0], interpolation=None)
        plt.figure()
        plt.imshow(pMitosis[:,:,1], interpolation=None)
        plt.figure()
        plt.imshow(pMitosis[:,:,2], interpolation=None)
        plt.figure()
        plt.imshow(pMitosis[:,:,0]/pMitosis[:,:,2], interpolation=None)
        plt.figure()
        plt.imshow(plt.imread(basename+".jpg"))
        plt.show()


def mitos12ConvAutoEncoder4(train=False, resumeTraining=False, iterations=20000):
    model = {
        'inputLayer' : ImageInputLayer(width=128,height=128,channels=3),
        'hiddenLayers' : [
            ConvolutionLayer(kernelsize=15, channels=3, features=12, stride=4, weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[128,128,3]),
            ConvolutionLayer(kernelsize=7, channels=12, features=40, stride=2,weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[32,32,12]),
            ConvolutionLayer(kernelsize=5, channels=40, features=80, stride=2,weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[16,16,40])
        ]
    }

    batch_size = 50
    net = Network(model, objective='reconstruction', batch_size=batch_size)
    net.setupTraining("squared-diff", "Adam", a=0.998)
    autoEncoderName = "mitos12ConvAutoEncoder4WithDistortion"

    init = tf.initialize_all_variables()
    basedir = "/media/sf_E_DRIVE/Dropbox/ULB/Doctorat/ImageSet/MITOS12/"
    mitos12 = MITOS12Data(train_dirs=[os.path.join(basedir,d) for d in ["A00_v2", "A01_v2", "A02_v2", "A03_v2", "A04_v2"]],chunksize=(128,128))

    saver = tf.train.Saver()
    sess = tf.Session()

    with sess.as_default():
        assert tf.get_default_session() is sess
        sess.run(init)
        if train==False or resumeTraining==True:
            saver.restore(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%autoEncoderName)
        if train==True:
            for i in range(iterations):
                batch_xs = mitos12.next_batch(batch_size, noise=True, nc=0.02)
                batch_ys = batch_xs
                
                if i%1000 == 0:
                    cost = net.cost.eval(feed_dict={net.x: batch_xs, net.target: batch_ys})
                    loss = net.loss.eval(feed_dict={net.x: batch_xs, net.target: batch_ys})
                    l2loss = net.l2loss.eval()
                    print "step %d, training cost %g, loss %g, l2loss %g"%(i,cost,loss,l2loss)
                    save_path = saver.save(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%autoEncoderName)
                    with open("/home/adrien/workspace/DeepNet/%s_results.txt"%autoEncoderName, "a") as resFile:
                        resFile.write("step %d, training cost %g, loss %g, l2loss %g\n"%(i,cost,loss,l2loss))
            
                net.train(batch_xs, batch_ys)

            save_path = saver.save(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%autoEncoderName)

        xs = mitos12.next_batch(batch_size)
        cost = net.cost.eval(feed_dict={net.x: xs, net.target: xs})
        loss = net.loss.eval(feed_dict={net.x: xs, net.target: xs})
        l2loss = net.l2loss.eval()
        print "test cost %g, loss %g, l2loss %g"%(cost,loss,l2loss)

        es = net.encode(xs)

        # for idx in range(40):
        W = net.layers[1].W.eval()
        W = (W-W.min())/(W.max()-W.min())
        plt.figure()
        for idx in range(12):
            plt.subplot(4,3,idx+1)
            plt.axis('off')
            plt.imshow(W[:,:,:,idx], interpolation='nearest', vmin=-1, vmax=1)
        # plt.show()
        # return
        
        # for idx in range(len(es)):
        #     max_act = 0
        #     arg_max_act = 0
        #     es0 = es[idx]
        #     for t in np.arange(0,160):
        #         if( es0[:,:,t].sum() > max_act ):
        #             max_act = es0[:,:,t].sum()
        #             arg_max_act = t
            
        #     print arg_max_act
        # return
        # featt = np.zeros(es0.shape)
        # featt[:,:,arg_max_act] = es0[:,:,arg_max_act]
        # rs = net.decode([featt])
        # print es0[:,:,arg_max_act]
        # plt.figure()
        # plt.gray()
        # plt.imshow(es0[:,:,arg_max_act], interpolation='nearest')
        # plt.figure()
        # plt.imshow(rs[0]/rs[0].max(), interpolation='nearest')
        # plt.show()
        # return

        rs = net.predict(xs)
        rs[0][rs[0]>1.] = 1.
        
        plt.figure()
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