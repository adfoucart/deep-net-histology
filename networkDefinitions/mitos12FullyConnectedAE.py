import tensorflow as tf
from NetworkLayer import *
from WeightInit import WeightInit
from MITOS12Data import MITOS12Data
from matplotlib import pyplot as plt
import numpy as np
from sys import stdout

from Network import Network

def mitos12FullyConnectedAE(train=False, resumeTraining=False, iterations=20000):
    model = {
        'inputLayer' : FlatInputLayer(inputsize=64*64*3),
        'hiddenLayers' : [
            FullyConnectedLayer(inputsize=64*64*3,outputsize=64*64,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh),
            FullyConnectedLayer(inputsize=64*64,outputsize=32*32,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh),
            FullyConnectedLayer(inputsize=32*32,outputsize=16*16,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh),
        ]
    }
    savedModelPath = "/home/adrien/workspace/DeepNet/mitos12FullyConnectedAE.ckpt"

    batch_size = 50
    net = Network(model, objective='reconstruction', batch_size=batch_size)
    net.setupTraining("squared-diff", "Adam", True)

    init = tf.initialize_all_variables()
    mitos12 = MITOS12Data(train_dirs=["/media/sf_VirtualDropbox"])

    saver = tf.train.Saver()
    sess = tf.Session()

    with sess.as_default():
        assert tf.get_default_session() is sess
        sess.run(init)
        if train==False or resumeTraining==True:
            print "Restoring from "+savedModelPath
            saver.restore(sess, savedModelPath)
        if train==True:
            for i in range(iterations):
                batch_xs = mitos12.next_batch(batch_size, flat=True)
                batch_ys = batch_xs
                
                if i%1000 == 0:
                    cost = net.cost.eval(feed_dict={net.x: batch_xs, net.target: batch_ys})
                    print "step %d, training cost %g"%(i,cost)
                    save_path = saver.save(sess,savedModelPath)
            
                net.train(batch_xs, batch_ys)

            save_path = saver.save(sess,savedModelPath)


        im = np.array(mitos12.getRandomImage())
        print im.shape
        cols = im.shape[0]/64
        rows = im.shape[1]/64
        xs_1 = [im[i*64:i*64+64,j*64:j*64+64,:].flatten() for i in xrange(cols) for j in xrange(rows)]
        xs_2 = [im[32+i*64:32+i*64+64,32+j*64:32+j*64+64,:].flatten() for i in xrange(cols) for j in xrange(rows)]
        rs_1 = net.predict(xs_1)
        rs_2 = net.predict(xs_2)
        rim = np.zeros(im.shape)
        im_k = np.zeros((im.shape[0],im.shape[1]))
        for k,r in enumerate(rs_1):
            i = (k/cols)*64
            j = (k%cols)*64
            rim[i:i+64,j:j+64,:] += r.reshape((64,64,3))
            im_k[i:i+64,j:j+64] += 1
        for k,r in enumerate(rs_2):
            i = (k/cols)*64
            j = (k%cols)*64
            rim[32+i:32+i+64,32+j:32+j+64,:] += r.reshape((64,64,3))
            im_k[32+i:32+i+64,32+j:32+j+64] += 1

        rim[im_k>0,0] /= im_k[im_k>0]
        rim[im_k>0,1] /= im_k[im_k>0]
        rim[im_k>0,2] /= im_k[im_k>0]
        rim[rim>1.] = 1.
        rim[rim<0.] = 0.
        plt.figure(0)
        plt.subplot(1,2,1)
        plt.imshow(im, interpolation='nearest')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(rim, interpolation='nearest')
        plt.axis('off')
        plt.show()

        # xs = mitos12.next_batch(batch_size, flat=True)
        # rs = net.predict(xs)
        # rs[0][rs[0]>1.] = 1.

        # xs = xs[0].reshape((64,64,3))
        # rs = rs[0].reshape((64,64,3))
        
        # plt.figure(1)
        # plt.subplot(2,3,1)
        # plt.title('Original')
        # plt.imshow(xs, interpolation='nearest')
        # plt.axis('off')
        # plt.subplot(2,3,2)
        # plt.title('Reconstruction')
        # plt.imshow(rs, interpolation='nearest')
        # plt.axis('off')
        # plt.gray()
        # plt.subplot(2,3,4)
        # plt.title('Diff - R')
        # plt.imshow(np.abs(rs-xs)[:,:,0], interpolation='nearest')
        # plt.axis('off')
        # plt.subplot(2,3,5)
        # plt.title('Diff - G')
        # plt.imshow(np.abs(rs-xs)[:,:,1], interpolation='nearest')
        # plt.axis('off')
        # plt.subplot(2,3,6)
        # plt.title('Diff - B')
        # plt.imshow(np.abs(rs-xs)[:,:,2], interpolation='nearest')
        # plt.axis('off')
        # plt.show()

def visuNetwork(n=0):
    model = {
        'inputLayer' : FlatInputLayer(inputsize=64*64*3),
        'hiddenLayers' : [
            FullyConnectedLayer(inputsize=64*64*3,outputsize=32*32,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh),
            FullyConnectedLayer(inputsize=32*32,outputsize=16*16,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh),
            FullyConnectedLayer(inputsize=16*16,outputsize=8*8,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh),
        ]
    }
    savedModelPath = "/home/adrien/workspace/DeepNet/mitos12FullyConnectedAE.ckpt"

    batch_size = 200
    net = Network(model, objective='reconstruction', batch_size=batch_size)
    net.setupTraining("squared-diff", "Adam", False)
    mitos12 = MITOS12Data(train_dirs=["/media/sf_VirtualDropbox"])

    varsToRestore = []
    for l in net.layers:
        varsToRestore += l.trainables
    saver = tf.train.Saver(varsToRestore)
    sess = tf.Session()

    init = tf.initialize_all_variables()

    with sess.as_default():
        assert tf.get_default_session() is sess
        sess.run(init)
        saver.restore(sess, savedModelPath)

        batch_xs = mitos12.next_batch(batch_size, flat=True)
        ys = net.encode(batch_xs)

        print ys.shape
        plt.figure(1)
        plt.imshow(ys, vmin=ys.min(), vmax=ys.max(), interpolation='nearest')
        plt.gray()
        plt.show()
        return

    #x = tf.Variable(tf.truncated_normal([1,64*64*3], stddev=0.1))
    # x = np.ones((1,64*64*3))*0.5

    # batch_size = 1
    # net = Network(model, objective='reconstruction', batch_size=batch_size)
    # net.setupTraining("squared-diff", "Adam", False)

    # varsToRestore = []
    # for l in net.layers:
    #     varsToRestore += l.trainables

    # mitos12 = MITOS12Data(train_dirs=["/media/sf_VirtualDropbox"])

    # saver = tf.train.Saver(varsToRestore)
    # sess = tf.Session()

    # init = tf.initialize_all_variables()

    # with sess.as_default():
    #     assert tf.get_default_session() is sess
    #     sess.run(init)
    #     saver.restore(sess, savedModelPath)

    #     W = net.layers[3].W.eval()
    #     print W.shape, W.min(), W.max()
    #     plt.figure(1)
    #     plt.imshow(W, vmin=W.min(), vmax=W.max())
    #     plt.gray()
    #     plt.show()
    #     return

    #     target = [0. for i in range(64)]
    #     target[n] = 1.
    #     print target

    #     y = net.encoded
    #     loss = y-target
    #     grad = tf.gradients(loss, net.x)

    #     for i in range(100):
    #         gradx = grad[0].eval(feed_dict={net.x: x})
    #         if i%100 == 0:
    #             print gradx
    #         x += gradx*0.1

    #     fig = plt.figure(1)
    #     plt.imshow(x.reshape([64,64,3]))
    #     fig.savefig('neuron_%d_max_input.png'%(n), bbox_inches='tight')
    #     print "Saved figure"