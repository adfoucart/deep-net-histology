import tensorflow as tf
from JPHOpenSlideAlguesData import JPHOpenSlideAlguesData
from MITOS12Data import MITOS12Data
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

def C(pred,target):
    C = np.zeros((2,2))
    for i in range(len(pred)):
        C[np.array(pred[i]).argmax(),np.array(target[i]).argmax()] += 1
    return C

class Network:

    def __init__(self, path, sess):
        self.setup(path, sess)

    def setup(self, path, sess):
        gd = tf.core.framework.graph_pb2.GraphDef()
        with open(path, "rb") as f:
            gd.ParseFromString(f.read())

        tf.import_graph_def(gd, name='inception5h')

        self.input = sess.graph.get_tensor_by_name('inception5h/input:0')

        # if inputshape != 256:
        #     self.input_ = tf.placeholder(tf.float32, [None, inputshape, inputshape, 3])
        #     self.resized = tf.image.resize_images(self.input_, 256, 256)
        # else:
        #     self.input_ = None

        self.features = sess.graph.get_tensor_by_name('inception5h/nn0/reshape:0')

        # Classif 1
        self.w1 = tf.Variable(tf.truncated_normal([1024,2], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([2]))
        # self.fc_1 = tf.nn.tanh(tf.matmul(self.features, self.w1)+self.b1)
        self.softmax = tf.nn.softmax(tf.matmul(self.features, self.w1)+self.b1)

        # Classif 2
        # self.w2 = tf.Variable(tf.truncated_normal([100,2], stddev=0.1))
        # self.b2 = tf.Variable(tf.zeros([2]))
        # self.softmax = tf.nn.softmax(tf.matmul(self.fc_1, self.w2)+self.b2)

    def get_features(self, X):
        # if self.input_ != None:
        #     return self.features.eval(feed_dict={self.input: self.resized.eval(feed_dict={self.input_:X})})
        return self.features.eval(feed_dict={self.input: X})

    def setupTraining(self, batch_size):
        self.target = tf.placeholder(tf.float32, [None, 2])

        lr = 1e-4
        a = 0.01
        self.cost = -tf.reduce_sum(self.target*tf.log(self.softmax))+a*(tf.nn.l2_loss(self.w1))#+tf.nn.l2_loss(self.w2))
        optimizer = tf.train.AdamOptimizer(lr)
        self.trainingStep = optimizer.minimize(self.cost, var_list=[self.w1, self.b1])#, self.w2, self.b2])

    def train(self, X, Y):
        # if self.input_ != None:
        #     tmp = self.resized.eval(feed_dict={self.input_: X})
        #     self.trainingStep.run(feed_dict={self.input: tmp, self.target: Y})
        self.trainingStep.run(feed_dict={self.input: X, self.target: Y})

    def predict(self, X):
        # if self.input_ != None:
        #     return self.softmax.eval(feed_dict={self.input: self.resized.eval(feed_dict={self.input_:X})})
        return self.softmax.eval(feed_dict={self.input: X})

def log(clf_name, msg):
    with open("%s_log.txt"%clf_name, "a") as f:
        f.write(msg+"\n")
    print msg

# data = JPHOpenSlideAlguesData(train_dirs=["/media/sf_E_DRIVE/data/media/images/"], chunksize=(256, 256), isfloat=True)
basedir = "/media/sf_E_DRIVE/Dropbox/ULB/Doctorat/ImageSet/MITOS12/"
data = MITOS12Data(train_dirs=[os.path.join(basedir,d) for d in ["A00_v2", "A01_v2", "A02_v2", "A03_v2", "A04_v2"]], chunksize=(256,256), resizeTo=(256,256))
inception_path = "inception5h/tensorflow_inception_graph.pb"

sess = tf.InteractiveSession()
net = Network(inception_path, sess)

clf_name = "mitos12_inception_clflayers_256"
saver_clf = tf.train.Saver([net.w1, net.b1])#, net.w2, net.b2])
batch_size = 100

net.setupTraining(batch_size)

tf.initialize_all_variables().run()

best_score = 100.

# saver_clf.restore(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_name)
saver_clf.restore(sess, "/home/adrien/workspace/DeepNet/%s_best.ckpt"%clf_name)

log(clf_name, "Start training sequence")
# for i in range(0,30000+1):
#     batch = data.next_supervised_batch(batch_size, noise=True, nc=0.02)
#     X = [b[0] for b in batch]
#     Y = [b[1] for b in batch]

#     net.train(X, Y)

#     if i%1000==0:
#         batch = data.next_supervised_batch(batch_size)
#         X = [b[0] for b in batch]
#         Y = [b[1] for b in batch]

#         cost = net.cost.eval(feed_dict={net.target: Y, net.input: X})
#         if cost < best_score:
#             best_score = cost
#             saver_clf.save(sess, "/home/adrien/workspace/DeepNet/%s_best.ckpt"%clf_name)
#             log(clf_name, "Saving current best classifier")
    
#         log(clf_name, "Iteration %d : Cost %g"%(i, cost))
#         w1 = net.w1.eval()
#         # w2 = net.w2.eval()
#         plt.figure()
#         # plt.subplot(1,2,1)
#         plt.imshow(w1, interpolation='none')
#         plt.axis('off')
#         plt.title('W1')
#         # plt.subplot(1,2,2)
#         # plt.imshow(w2, interpolation='none')
#         # plt.axis('off')
#         # plt.title('W2')
#         plt.savefig('weights_for_%s_%i.png'%(clf_name, i))
#         plt.close()

#         # Eval :
#         # Cmat = np.zeros((2,2))
#         # for i in range(50):
#         #     batch = data.next_supervised_batch(batch_size)
#         #     X = [b[0] for b in batch]
#         #     Y = [b[1] for b in batch]
            
#         #     pred = net.predict(X)
#         #     Cmat += C(pred,Y)

#         # print Cmat

#         saver_clf.save(sess, "/home/adrien/workspace/DeepNet/%s.ckpt"%clf_name)

basename = "/media/sf_E_DRIVE/Dropbox/ULB/Doctorat/ImageSet/MITOS12/A_eval/A01_v2/A01_04"
im_ = Image.open(basename+".bmp")
im_, path, basename = data.images[18]
im = np.array(im_)
stride = 32
rangex = np.arange(0,im.shape[0]-data.chunksize[0],stride)
rangey = np.arange(0,im.shape[1]-data.chunksize[1],stride)
ts = [(t/len(rangey), t%len(rangey)) for t in range(len(rangex)*len(rangey))]
chunks = [np.asarray(im_.crop((tx*stride, ty*stride, tx*stride+data.chunksize[0], ty*stride+data.chunksize[1])).resize(data.resizeTo)).astype(np.float32)/255. for tx,ty in ts]
# chunks = [im[tx*stride:tx*stride+data.chunksize[0],ty*stride:ty*stride+data.chunksize[1],:].astype(np.float32)/255. for tx,ty in ts]
chunksPos = [(tx*stride,ty*stride) for tx,ty in ts]
pMitosis = np.zeros((im.shape[0], im.shape[1], 3))

print len(chunks)
for t in range(len(chunks)/50):
    batch = chunks[t*50:t*50+50]
    is_mitosis = net.predict(batch)
    for i,p in enumerate(is_mitosis):
        cp = chunksPos[t*50+i]
        pMitosis[cp[0]:cp[0]+data.chunksize[0], cp[1]:cp[1]+data.chunksize[1], 0] += p[0]
        pMitosis[cp[0]:cp[0]+data.chunksize[0], cp[1]:cp[1]+data.chunksize[1], 1] = np.maximum(pMitosis[cp[0]:cp[0]+data.chunksize[0], cp[1]:cp[1]+data.chunksize[1],1], pMitosis[cp[0]:cp[0]+data.chunksize[0], cp[1]:cp[1]+data.chunksize[1],0])
        pMitosis[cp[0]:cp[0]+data.chunksize[0], cp[1]:cp[1]+data.chunksize[1], 2] += 1

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

# batch = data.next_batch(300, noise=True, nc=0.02)
# feats = net.get_features(batch)
# mean_feats = feats.mean(axis=0)
# std_feats = feats.std(axis=0)

# feats_ = (feats-mean_feats)/(std_feats)

# from matplotlib import pyplot as plt

# plt.figure()
# plt.imshow(feats, interpolation='none', cmap=plt.cm.gray)
# plt.figure()
# plt.imshow(feats_, interpolation='none', cmap=plt.cm.gray)
# plt.show()