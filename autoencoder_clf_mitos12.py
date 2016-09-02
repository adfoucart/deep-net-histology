import tensorflow as tf
from MITOS12Data import MITOS12Data
import os
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
        self.input = tf.placeholder(tf.float32, [None,128,128,3])

        # Convolution 1
        self.w1 = tf.Variable(tf.truncated_normal([15,15,3,16], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([16]))
        self.conv_1 = tf.nn.tanh(tf.nn.conv2d(self.input, self.w1, strides=[1, 2, 2, 1], padding='SAME')+self.b1)

        # Convolution 2
        self.w2 = tf.Variable(tf.truncated_normal([7,7,16,32], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([32]))
        self.conv_2 = tf.nn.tanh(tf.nn.conv2d(self.conv_1, self.w2, strides=[1, 2, 2, 1], padding='SAME')+self.b2)

        # Convolution 3
        self.w3 = tf.Variable(tf.truncated_normal([5,5,32,32], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([32]))
        self.conv_3 = tf.nn.tanh(tf.nn.conv2d(self.conv_2, self.w3, strides=[1, 2, 2, 1], padding='SAME')+self.b3)

        # Convolution 4
        self.w4 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
        self.b4 = tf.Variable(tf.zeros([64]))
        self.conv_4 = tf.nn.tanh(tf.nn.conv2d(self.conv_3, self.w4, strides=[1, 2, 2, 1], padding='SAME')+self.b4)

        # Vectorize
        self.vec_4 = tf.reshape(self.conv_4, [-1,8*8*64])

        # Classif 1
        self.w5 = tf.Variable(tf.truncated_normal([8*8*64,200], stddev=0.1))
        self.b5 = tf.Variable(tf.zeros([200]))
        self.fc_1 = tf.nn.tanh(tf.matmul(self.vec_4, self.w5)+self.b5)
        # Classif 2
        self.w6 = tf.Variable(tf.truncated_normal([200,2], stddev=0.1))
        self.b6 = tf.Variable(tf.zeros([2]))
        self.softmax = tf.nn.softmax(tf.matmul(self.fc_1, self.w6)+self.b6)

        # Outputs
        self.y = self.conv_4

        return self.y

    def encode(self, x):
        return self.y.eval(feed_dict={self.input: x})

    def get_decoder(self):
        # Deconvolution 1
        deconv_4 = lambda e,n : tf.nn.tanh(tf.nn.deconv2d(e, self.w4, [n, 16, 16, 32], strides=[1,2,2,1], padding="SAME"))
        deconv_3 = lambda e,n : tf.nn.tanh(tf.nn.deconv2d(e, self.w3, [n, 32, 32, 32], strides=[1,2,2,1], padding="SAME"))
        deconv_2 = lambda e,n : tf.nn.tanh(tf.nn.deconv2d(e, self.w2, [n, 64, 64, 16], strides=[1,2,2,1], padding="SAME"))
        deconv_1 = lambda e,n : tf.nn.tanh(tf.nn.deconv2d(e, self.w1, [n,128,128,3], strides=[1, 2, 2, 1], padding="SAME"))

        return deconv_4,deconv_3,deconv_2,deconv_1

    def decode(self, x):
        e = tf.placeholder(tf.float32, [None,8,8,64])

        deconv_4, deconv_3, deconv_2, deconv_1 = self.get_decoder()
        self.decoded = deconv_1(deconv_2(deconv_3(deconv_4(e, len(x)),len(x)), len(x)), len(x))
        # self.decoded = deconv_1(deconv_2(deconv_3(deconv_4(e, len(x)), len(x)), len(x)), len(x))
        return self.decoded.eval(feed_dict={e: x})

    def setupClassifTraining(self, batch_size):
        self.target = tf.placeholder(tf.float32, [None, 2])

        lr = 1e-4
        a = 0.01
        self.cost = -tf.reduce_sum(self.target*tf.log(self.softmax))#+a*(tf.nn.l2_loss([self.w5,self.w6]))
        optimizer = tf.train.AdamOptimizer(lr)
        self.trainingStep = optimizer.minimize(self.cost, var_list=[self.w5, self.b5, self.w6, self.b6])

    def setupTraining(self, batch_size):
        self.target = tf.placeholder(tf.float32, self.input.get_shape())

        deconv_4, deconv_3, deconv_2, deconv_1 = self.get_decoder()
        self.decoded = deconv_1(deconv_2(deconv_3(deconv_4(self.y, batch_size), batch_size), batch_size), batch_size)
        # self.decoded = deconv_1(deconv_2(deconv_3(deconv_4(self.y, batch_size), batch_size), batch_size), batch_size)
        # self.decoded = self.get_decoder()(self.y,batch_size)

        lr = 1e-4
        a = 0.
        self.cost = tf.reduce_mean(tf.square(self.target-self.decoded))+a*(tf.nn.l2_loss(self.w4))
        optimizer = tf.train.AdamOptimizer(lr)
        self.trainingStep = optimizer.minimize(self.cost, var_list=[self.w4, self.b4])

    def train(self, X, Y):
        self.trainingStep.run(feed_dict={self.input: X, self.target: Y})

    def predict(self, X):
        return self.softmax.eval(feed_dict={self.input: X})

basedir = "/media/sf_E_DRIVE/Dropbox/ULB/Doctorat/ImageSet/MITOS12/"
mitos12 = MITOS12Data(train_dirs=[os.path.join(basedir,d) for d in ["A00_v2", "A01_v2", "A02_v2", "A03_v2", "A04_v2"]], chunksize=(128,128))

# Get test image
test_image = mitos12.next_batch(1)
clf_name_1 = "mitos12_ae_conv_1layer"
clf_name_2 = "mitos12_ae_conv_2layer"
clf_name_3 = "mitos12_ae_conv_3layer"
clf_name_4 = "mitos12_ae_conv_4layer"
clf_name_5 = "mitos12_ae_conv_clflayers"

net = Network()
saver_1 = tf.train.Saver([net.w1, net.b1])
saver_2 = tf.train.Saver([net.w2, net.b2])
saver_3 = tf.train.Saver([net.w3, net.b3])
saver_4 = tf.train.Saver([net.w4, net.b4])
saver_5 = tf.train.Saver([net.w5, net.b5, net.w6, net.b6])
sess = tf.InteractiveSession()
print "Started Tensorflow session"

batch_size = 100
net.setupClassifTraining(batch_size)

tf.initialize_all_variables().run()

saver_1.restore(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_name_1)
saver_2.restore(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_name_2)
saver_3.restore(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_name_3)
saver_4.restore(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_name_4)
saver_5.restore(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_name_5)

'''w2 = net.w2.eval()
w2s = w2.reshape((7*7*16,32))

plt.figure()
plt.gray()
plt.imshow(w2s)
plt.axis('off')
plt.colorbar()
plt.show()'''

'''for k in range(32):
    e = np.zeros((1,32,32,32))
    e[:,:,:,k] = 1
    d = net.decode(e)
    d = (d-d.min())/(d.max()-d.min())
    plt.figure()
    plt.imshow(d[0])
    plt.axis('off')
    plt.savefig("mitos12_ae_results/mitos12_conv_maxforlastlayer_autoencoder_%04d.png"%k)
    plt.close()'''

'''e = net.encode(test_image)
d = net.decode(e)

print "Start training sequence"
for i in range(20000):
    # batch = mitos12.next_batch(batch_size, noise=True, nc=0.02)
    batch = mitos12.next_supervised_batch(batch_size, noise=True, nc=0.02)
    X = [b[0] for b in batch]
    Y = [b[1] for b in batch]
    net.train(X, Y)

    if i%1000==0:
        # e = net.encode(test_image)
        # d = net.decode(e)
        # d[d>1.] = 1.

        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(test_image[0])
        # plt.axis('off')
        # plt.subplot(1,2,2)
        # plt.imshow(d[0])
        # plt.axis('off')
        # plt.savefig("mitos12_ae_results/mitos12_conv_4layer_autoencoder_%04d.png"%(20000+i))
        # plt.close()

        # Ws = net.w1.eval()
        # Ws = (Ws-Ws.min())/(Ws.max()-Ws.min())
        # plt.figure()
        # for j in range(16):
        #     plt.subplot(4,4,j+1)
        #     plt.imshow(Ws[:,:,:,j])
        #     plt.axis('off')
        # plt.savefig("mitos12_ae_results/mitos12_conv_3layer_autoencoder_weights_%04d.png"%i)
        # plt.close()

        print "Iteration %d : Cost %g"%(i, net.cost.eval(feed_dict={net.target: Y, net.input: X}))

        # Eval :
        Cmat = np.zeros((2,2))
        for i in range(50):
            batch = mitos12.next_supervised_batch(batch_size)
            X = [b[0] for b in batch]
            Y = [b[1] for b in batch]
            
            pred = net.predict(X)
            Cmat += C(pred,Y)

        print Cmat

        #saver_1.save(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_name_1)
        #saver_2.save(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_name_2)
        #saver_3.save(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_name_3)
        #saver_4.save(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_name_4)
        #saver_5.save(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_name_5)'''

im_, path, basename = mitos12.images[18]
im = np.array(im_)
stride = 15
rangex = np.arange(0,im.shape[0]-128,stride)
rangey = np.arange(0,im.shape[1]-128,stride)
ts = [(t/len(rangey), t%len(rangey)) for t in range(len(rangex)*len(rangey))]
chunks = [im[tx*stride:tx*stride+128,ty*stride:ty*stride+128,:].astype(np.float32)/255. for tx,ty in ts]
chunksPos = [(tx*stride,ty*stride) for tx,ty in ts]
pMitosis = np.zeros((im.shape[0], im.shape[1], 3))

print len(chunks)
for t in range(len(chunks)/50):
    batch = chunks[t*50:t*50+50]
    is_mitosis = net.predict(batch)
    for i,p in enumerate(is_mitosis):
        cp = chunksPos[t*50+i]
        pMitosis[cp[0]:cp[0]+128, cp[1]:cp[1]+128, 0] += p[0]
        pMitosis[cp[0]:cp[0]+128, cp[1]:cp[1]+128, 1] = np.maximum(pMitosis[cp[0]:cp[0]+128, cp[1]:cp[1]+128,1], pMitosis[cp[0]:cp[0]+128, cp[1]:cp[1]+128,0])
        pMitosis[cp[0]:cp[0]+128, cp[1]:cp[1]+128, 2] += 1

plt.figure()
plt.gray()
plt.imshow(pMitosis[:,:,0], interpolation="None")
plt.figure()
plt.imshow(pMitosis[:,:,1], interpolation="None")
plt.figure()
plt.imshow(pMitosis[:,:,2], interpolation="None")
plt.figure()
plt.imshow(pMitosis[:,:,0]/pMitosis[:,:,2], interpolation="None")
plt.figure()
plt.imshow(plt.imread(basename+".jpg"))
plt.show()