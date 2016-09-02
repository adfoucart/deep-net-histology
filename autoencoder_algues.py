import tensorflow as tf
from JPHOpenSlideAlguesData import JPHOpenSlideAlguesData
import os
from os import path
from matplotlib import pyplot as plt
import numpy as np

def C(pred,target):
    C = np.zeros((2,2))
    for i in range(len(pred)):
        C[np.array(pred[i]).argmax(),np.array(target[i]).argmax()] += 1
    return C

class Network:
    def __init__(self):
        self.setup()
    
    def setup(self):
        # INPUT
        self.input = tf.placeholder(tf.float32, [None,256,256,3])
        self.layer_shapes = [ [256, 256, 3], [128, 128, 16], [64, 64, 32], [32, 32, 32], [16, 16, 64] ]

        # Convolution 1
        self.w1 = tf.Variable(tf.truncated_normal([15,15,self.layer_shapes[0][2],self.layer_shapes[1][2]], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([16]))
        self.conv_1 = tf.nn.tanh(tf.nn.conv2d(self.input, self.w1, strides=[1, 2, 2, 1], padding='SAME')+self.b1)

        # Convolution 2
        self.w2 = tf.Variable(tf.truncated_normal([7,7,self.layer_shapes[1][2],self.layer_shapes[2][2]], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([32]))
        self.conv_2 = tf.nn.tanh(tf.nn.conv2d(self.conv_1, self.w2, strides=[1, 2, 2, 1], padding='SAME')+self.b2)

        # Convolution 3
        self.w3 = tf.Variable(tf.truncated_normal([5,5,self.layer_shapes[2][2],self.layer_shapes[3][2]], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([32]))
        self.conv_3 = tf.nn.tanh(tf.nn.conv2d(self.conv_2, self.w3, strides=[1, 2, 2, 1], padding='SAME')+self.b3)

        # Convolution 4
        self.w4 = tf.Variable(tf.truncated_normal([5,5,self.layer_shapes[3][2],self.layer_shapes[4][2]], stddev=0.1))
        self.b4 = tf.Variable(tf.zeros([64]))
        self.conv_4 = tf.nn.tanh(tf.nn.conv2d(self.conv_3, self.w4, strides=[1, 2, 2, 1], padding='SAME')+self.b4)

        self.outputs = [self.conv_1, self.conv_2, self.conv_3, self.conv_4]
        self.trainables = [ [self.w1, self.b1], [self.w2, self.b2], [self.w3, self.b3], [self.w4, self.b4] ]

        return self.outputs

    def encode(self, x, layer=0):
        return self.outputs[layer].eval(feed_dict={self.input: x})

    def get_decoder(self):
        # Deconvolution 1
        deconv_4 = lambda e,n : tf.nn.tanh(tf.nn.deconv2d(e, self.w4, [n,self.layer_shapes[3][0],self.layer_shapes[3][1],self.layer_shapes[3][2]], strides=[1,2,2,1], padding="SAME"))
        deconv_3 = lambda e,n : tf.nn.tanh(tf.nn.deconv2d(e, self.w3, [n,self.layer_shapes[2][0],self.layer_shapes[2][1],self.layer_shapes[2][2]], strides=[1,2,2,1], padding="SAME"))
        deconv_2 = lambda e,n : tf.nn.tanh(tf.nn.deconv2d(e, self.w2, [n,self.layer_shapes[1][0],self.layer_shapes[1][1],self.layer_shapes[1][2]], strides=[1,2,2,1], padding="SAME"))
        deconv_1 = lambda e,n : tf.nn.tanh(tf.nn.deconv2d(e, self.w1, [n,self.layer_shapes[0][0],self.layer_shapes[0][1],self.layer_shapes[0][2]], strides=[1,2,2,1], padding="SAME"))

        return [deconv_1,deconv_2,deconv_3,deconv_4]

    def decode(self, x, layer=None):
        if layer == None:
            layer = len(deconv_layers)

        e = tf.placeholder(tf.float32, [None,self.layer_shapes[layer+1][0],self.layer_shapes[layer+1][1],self.layer_shapes[layer+1][2]])

        deconv_layers = self.get_decoder()

        cur = e
        for l in range(0, layer+1):
            cur = deconv_layers[layer-l](cur, len(x))

        self.decoded = cur
        
        return self.decoded.eval(feed_dict={e: x})

    def setupTraining(self, batch_size, layer=None):
        self.target = tf.placeholder(tf.float32, self.input.get_shape())
        
        deconv_layers = self.get_decoder()
        if layer == None:
            layer = len(deconv_layers)

        cur = self.outputs[layer]
        for l in range(0, layer+1):
            cur = deconv_layers[layer-l](cur, batch_size)

        self.decoded = cur

        lr = 1e-4
        a = 0.0
        self.cost = tf.reduce_mean(tf.square(self.target-self.decoded))+a*(tf.nn.l2_loss(self.w2))
        optimizer = tf.train.AdamOptimizer(lr)
        self.trainingStep = optimizer.minimize(self.cost, var_list=self.trainables[layer])

    def train(self, X, Y):
        self.trainingStep.run(feed_dict={self.input: X, self.target: Y})

    def predict(self, X):
        return self.softmax.eval(feed_dict={self.input: X})


data = JPHOpenSlideAlguesData(train_dirs=["/media/sf_E_DRIVE/data/media/images/"], chunksize=(256, 256))

clf_basename = "jph_ae_conv_%dlayer"
clf_names = [clf_basename%(i+1) for i in range(4)]
# clf_name_1 = clf_basename%1
# clf_name_2 = clf_basename%2
# clf_name_3 = clf_basename%3
# clf_name_4 = clf_basename%4

train_level = 3
clf_cur_name = clf_names[train_level-1]

net = Network()
saver_1 = tf.train.Saver([net.w1, net.b1])
saver_2 = tf.train.Saver([net.w2, net.b2])
saver_3 = tf.train.Saver([net.w3, net.b3])
saver_4 = tf.train.Saver([net.w4, net.b4])
savers = [saver_1, saver_2, saver_3, saver_4]

sess = tf.InteractiveSession()
print "Started Tensorflow session"

batch_size = 100
net.setupTraining(batch_size, train_level-1)

tf.initialize_all_variables().run()
for i in range(train_level):
    if os.path.isfile("/home/adrien/workspace/DeepNet/%s.ckpt"%clf_names[i]):
        savers[i].restore(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_names[i])

# saver_1.restore(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_name_1)
# saver_2.restore(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_name_2)
# saver_3.restore(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_name_3)
# saver_4.restore(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_name_4)

print "Start training sequence"
for i in range(10000):
    # batch = data.next_batch(batch_size, noise=True, nc=0.02)
    # net.train(batch, batch)

    if i%1000==0:
        batch = data.next_batch(1, noise=True, nc=0.02)
        e = net.encode([batch[0]], 2)
        d = net.decode(e, 2)
        d[d>1.] = 1.

        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(batch[0])
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(d[0])
        plt.axis('off')
        plt.savefig("jph_ae_results/%s__%04d.png"%(clf_cur_name, i))
        plt.close()

        # Ws = net.w1.eval()
        # Ws = (Ws-Ws.min())/(Ws.max()-Ws.min())
        # plt.figure()
        # for j in range(16):
        #     plt.subplot(4,4,j+1)
        #     plt.imshow(Ws[:,:,:,j])
        #     plt.axis('off')
        # plt.savefig("mitos12_ae_results/mitos12_conv_3layer_autoencoder_weights_%04d.png"%i)
        # plt.close()

        # batch = data.next_batch(batch_size, noise=True, nc=0.02)
        # print "Iteration %d : Cost %g"%(i, net.cost.eval(feed_dict={net.target: batch, net.input: batch}))

        # savers[train_level-1].save(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_cur_name)

