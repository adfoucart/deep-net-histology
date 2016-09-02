import tensorflow as tf

class Layer():
    def __init__(self, listvars, listargs):
        self.vars = {}
        for v in listvars:
            self.vars[v] = listargs[v] if v in listargs else None
        self.trainables = []

class FlatInputLayer(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, ['inputsize'], kwargs)
        self.x = tf.placeholder(tf.float32, [None, self.vars['inputsize']])
    def link(self):
        self.h = self.x

class ImageInputLayer(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, ['height', 'width', 'channels'], kwargs)
        self.x = tf.placeholder(tf.float32, [None,self.vars['height'],self.vars['width'],self.vars['channels']])
    def link(self):
        self.h = self.x

class ImageToVectorLayer(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, ['imagesize'], kwargs)
    def link(self, x):
        self.h = tf.reshape(x, [-1,self.vars['imagesize'][0]*self.vars['imagesize'][1]*self.vars['imagesize'][2]])

class VectorToImageLayer(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, ['imagesize'], kwargs)
    def link(self, x):
        self.h = tf.reshape(x, [-1, self.vars['imagesize'][0], self.vars['imagesize'][1], self.vars['imagesize'][2]])
        
class FullyConnectedLayer(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, ['inputsize', 'outputsize', 'weightInitFunc', 'biasInitFunc', 'activationFunc'], kwargs)
        self.W = self.vars['weightInitFunc']([self.vars['inputsize'], self.vars['outputsize']])
        self.b = self.vars['biasInitFunc']([self.vars['outputsize']])
        tf.add_to_collection('l2losses', tf.nn.l2_loss(self.W))
        # tf.add_to_collection('l2losses', tf.nn.l2_loss(self.b))
        self.invertedBy = InvertedLayer
        self.trainables = [self.W, self.b]
    def link(self, x):
        self.h = self.vars['activationFunc'](tf.matmul(x,self.W)+self.b)

class InvertedLayer(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, ['layer'], kwargs)
        self.W = tf.transpose(self.vars['layer'].W)
        self.b = tf.constant(0., shape=[self.vars['layer'].W.get_shape()[0].value])
    def link(self, x):
        self.h = self.vars['layer'].vars['activationFunc'](tf.matmul(x,self.W)+self.b)

class ConvolutionLayer(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, ['kernelsize', 'channels', 'features', 'stride', 'weightInitFunc', 'biasInitFunc', 'activationFunc', 'inputshape'], kwargs)
        self.W = self.vars['weightInitFunc']([self.vars['kernelsize'],self.vars['kernelsize'],self.vars['channels'],self.vars['features']])
        self.b = self.vars['biasInitFunc']([self.vars['features']])
        tf.add_to_collection('l2losses', tf.nn.l2_loss(self.W))
        # tf.add_to_collection('l2losses', tf.nn.l2_loss(self.b))
        self.invertedBy = DeconvolutionLayer
        self.trainables = [self.W, self.b]
    def link(self, x):
        self.h = self.vars['activationFunc'](tf.nn.conv2d(x, self.W, strides=[1, self.vars['stride'], self.vars['stride'], 1], padding='SAME')+self.b)

class DeconvolutionLayer(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, ['layer', 'batch_size'], kwargs)
        self.W = self.vars['layer'].W
        self.b = tf.constant(0., shape=[self.vars['layer'].W.get_shape()[0].value])
        self.activationFunc = self.vars['layer'].vars['activationFunc']
        self.stride = self.vars['layer'].vars['stride']
    def link(self, x):
        outputshape = [self.vars['batch_size']] + self.vars['layer'].vars['inputshape']
        self.h = self.activationFunc(tf.nn.deconv2d(x, self.W, outputshape, strides=[1, self.stride, self.stride, 1], padding="SAME"))

class MaxPoolingLayer(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, ['poolingSize'], kwargs)
    def link(self, x):
        self.h = tf.nn.max_pool(x, ksize=[1, self.vars['poolingSize'], self.vars['poolingSize'], 1], strides=[1,self.vars['poolingSize'],self.vars['poolingSize'],1], padding="SAME")