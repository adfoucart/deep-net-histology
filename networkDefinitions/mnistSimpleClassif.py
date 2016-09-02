import tensorflow as tf
from NetworkLayer import *
from WeightInit import WeightInit
from MNISTData import MNISTData
from matplotlib import pyplot as plt
import numpy as np
from sys import stdout

from Network import Network

def mnistSimpleClassif(train=False):
    model = {
        'inputLayer' : FlatInputLayer(inputsize=784),
        'hiddenLayers' : [
            FullyConnectedLayer(inputsize=784,outputsize=100,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh),
            ],
        'outputLayer' : FullyConnectedLayer(inputsize=100,outputsize=10,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.softmax)
    }

    batch_size = 50
    net = Network(model, objective='classification', batch_size=batch_size)
    net.setupTraining("cross-entropy", "Adam", wd=False)

    init = tf.initialize_all_variables()
    mnist = MNISTData(train_dir='MNIST_data', one_hot=True)

    saver = tf.train.Saver()
    sess = tf.Session()

    with sess.as_default():
        assert tf.get_default_session() is sess
        sess.run(init)
        if train==False:
            saver.restore(sess, "/home/adrien/workspace/DeepNet/mnistSimpleClassif.ckpt")
            print "Test accuracy %g"%net.evaluate(mnist.test['images'], mnist.test['labels'])
        else:
            for i in range(20000):
                batch_xs, batch_ys = mnist.next_batch(batch_size, set=mnist.train)
                
                if i%1000 == 0:
                    cost = net.cost.eval(feed_dict={net.x: batch_xs, net.target: batch_ys})
                    print "step %d, training cost %g"%(i,cost)
            
                net.train(batch_xs, batch_ys)

            print "Test accuracy %g"%net.evaluate(mnist.test['images'], mnist.test['labels'])
            save_path = saver.save(sess, "/home/adrien/workspace/DeepNet/mnistSimpleClassif.ckpt")

        plt.figure(1)
        plt.gray()
        plt.imshow(mnist.test['images'][0].reshape((28,28)), interpolation='nearest')
        plt.title(str(np.argmax(net.predict([mnist.test['images'][0]]))))
        plt.show()