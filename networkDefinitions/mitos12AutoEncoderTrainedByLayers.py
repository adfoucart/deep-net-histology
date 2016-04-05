import tensorflow as tf
from NetworkLayer import *
from WeightInit import WeightInit
from MITOS12Data import MITOS12Data
from matplotlib import pyplot as plt
import numpy as np
from sys import stdout

from Network import Network

def mitos12AutoEncoderTrainedByLayers(train=False, trainLayer=0, resumeTraining=False, iterations=20000):
    savedModelPath = "/home/adrien/workspace/DeepNet/mitos12ConvAutoEncoderTrainedByLayers.ckpt"

    print "Initializing model"
    firstLayerModel = {
        'inputLayer': ImageInputLayer(width=128,height=128,channels=3),
        'hiddenLayers' : [
            ConvolutionLayer(kernelsize=5, channels=3, features=32, stride=2, weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[128,128,3]),
        ]
    }
    secondLayerModel = {
        'inputLayer': ImageInputLayer(width=64,height=64,channels=32),
        'hiddenLayers' : [
            ConvolutionLayer(kernelsize=5, channels=32, features=64, stride=2, weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[64,64,32]),
        ]
    }
    thirdLayerModel = {
        'inputLayer': ImageInputLayer(width=32,height=32,channels=64),
        'hiddenLayers' : [
            ConvolutionLayer(kernelsize=5, channels=64, features=128, stride=2, weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[32,32,64]),
        ]
    }
    
    batch_size = 50

    models = [firstLayerModel, secondLayerModel, thirdLayerModel]
    nets = [Network(model, objective='reconstruction', batch_size=batch_size) for model in models]
    for net in nets:
        net.setupTraining("squared-diff", "Adam")

    init = tf.initialize_all_variables()
    mitos12 = MITOS12Data(train_dirs=["/media/sf_VirtualDropbox"], chunksize=(128,128))

    saver = tf.train.Saver()
    sess = tf.Session()

    with sess.as_default():
        assert tf.get_default_session() is sess
        sess.run(init)

        if train==False or resumeTraining==True:
            print "Restoring from "+savedModelPath
            saver.restore(sess, savedModelPath)

        # vizualizeConv(nets)
        # return
        
        if train==True:
            assert trainLayer >= 0 and trainLayer < len(nets)
            print "Training for "+str(iterations)+" iterations and "+str(trainLayer+1)+" layers"

            for i in range(iterations):
                batch_xs = mitos12.next_batch(batch_size)
                l = 0
                while l < trainLayer:
                    batch_xs = nets[l].encode(batch_xs)
                    l += 1

                batch_ys = batch_xs
                net = nets[trainLayer]
                if i%1000 == 0:
                    cost = net.cost.eval(feed_dict={net.x: batch_xs, net.target: batch_ys})
                    print "step %d, training cost %g"%(i,cost)
                    saver.save(sess, savedModelPath)
            
                net.train(batch_xs, batch_ys)
            
            save_path = saver.save(sess, savedModelPath)

            # Show result for the trained layer
            xs = mitos12.next_batch(1)        
            
            plt.figure(1)
            plt.subplot(2,3,1)
            plt.title('Original')
            plt.imshow(xs[0], interpolation='nearest')
            plt.axis('off')

            l = 0
            current = xs
            while l < trainLayer:
                print "Encoding layer "+str(l)
                current = nets[l].encode(current)
                l += 1
            es = nets[trainLayer].encode(current)
            #rs = nets[trainLayer].decode(es)
            rs = es
            while l >= 0:
                print "Decoding layer "+str(l)
                rs = nets[l].decode(rs)
                #rs[0][rs[0]>1.] = 1.
                l -= 1
            rs[0][rs[0]>1.] = 1.

            plt.subplot(2,3,2)
            plt.title('Reconstruction from layer '+str(trainLayer))
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
        else:
            # Show result for the last layer
            original = mitos12.next_batch(1)        
            
            plt.figure(1)
            plt.subplot(2,3,1)
            plt.title('Original')
            plt.imshow(original[0], interpolation='nearest')
            plt.axis('off')

            l = 0
            xs = original
            while l < len(nets):
                xs = nets[l].encode(xs)
                l += 1
            rs = xs
            while l > 0:
                l -= 1
                rs = nets[l].decode(rs)
                rs[0][rs[0]>1.] = 1.

            plt.subplot(2,3,2)
            plt.title('Reconstruction from all layer')
            plt.imshow(rs[0], interpolation='nearest')
            plt.axis('off')
            plt.gray()
            plt.subplot(2,3,4)
            plt.title('Diff - R')
            plt.imshow(np.abs(rs[0]-original[0])[:,:,0], interpolation='nearest')
            plt.axis('off')
            plt.subplot(2,3,5)
            plt.title('Diff - G')
            plt.imshow(np.abs(rs[0]-original[0])[:,:,1], interpolation='nearest')
            plt.axis('off')
            plt.subplot(2,3,6)
            plt.title('Diff - B')
            plt.imshow(np.abs(rs[0]-original[0])[:,:,2], interpolation='nearest')
            plt.axis('off')
            plt.show()