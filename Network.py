import tensorflow as tf
import cPickle
from NetworkLayer import InvertedLayer

class Network():
    def __init__(self, networkDefinition, **kwargs):
        print networkDefinition
        inputLayer = networkDefinition['inputLayer']
        hiddenLayers = networkDefinition['hiddenLayers']

        if 'objective' in kwargs:
            self.objective = kwargs['objective']
        else:
            self.objective = 'classification'

        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        else:
            self.batch_size = None

        self.layers = []
        self.x = inputLayer.x

        inputLayer.link()
        self.layers.append(inputLayer)
        
        curx = inputLayer.h
        self.summaries = []
        for idl,hl in enumerate(hiddenLayers):
            hl.link(curx)
            self.layers.append(hl)
            curx = hl.h
            if hasattr(hl, 'W') and hasattr(hl, 'b') and hasattr(hl, 'h'):
                self.summaries += [tf.histogram_summary("W"+str(idl), hl.W), tf.histogram_summary("b"+str(idl), hl.b), tf.histogram_summary("h"+str(idl), hl.h)]
            print curx

        if( self.objective == 'classification' ):
            outputLayer = networkDefinition['outputLayer']
            outputLayer.link(curx)
            self.layers.append(outputLayer)
            self.y = outputLayer.h
        elif( self.objective == 'reconstruction' ) :
            self.encoded = curx
            for i in range(len(self.layers)-1):
                l = self.layers[len(self.layers)-i-1]
                decodingLayer = l.invertedBy(layer=l, batch_size=self.batch_size)
                decodingLayer.link(curx)
                curx = decodingLayer.h
                print curx
            self.y = curx

    def setupTraining(self, costFunction="cross-entropy", optimizer="Adam", wd=True, trainVars=None, **kwargs):
        self.target = tf.placeholder(tf.float32, self.y.get_shape(), "Target")

        a = kwargs["a"] if "a" in kwargs else 0.99
        lr = kwargs["lr"] if "lr" in kwargs else 1e-4

        if costFunction == "cross-entropy":
            self.loss = -tf.reduce_sum(self.target*tf.log(self.y))
        elif costFunction == "squared-diff":
            self.loss = tf.reduce_mean(tf.square(self.target-self.y))

        if wd == True:
            # a = 0.99
            l2loss = tf.get_collection('l2losses')
            self.l2loss = tf.add_n(l2loss)/len(l2loss)
            self.cost = a*self.loss+(1-a)*self.l2loss
        else:
            self.cost = self.loss
            self.l2loss = tf.constant(0.)

        varsToTrain = []
        if trainVars == None :
            for l in self.layers:
                varsToTrain += l.trainables
        else:
            varsToTrain = trainVars

        if optimizer == "Adam":
            self.trainingStep = tf.train.AdamOptimizer(lr).minimize(self.cost, var_list=varsToTrain)
        elif optimizer == "GradientDescent":
            self.trainingStep = tf.train.GradientDescentOptimizer(lr).minimize(self.cost, var_list=varsToTrain)
        
        if(self.objective == 'classification' ):
            self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.target,1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        elif( self.objective == 'reconstruction' ):
            self.accuracy = (1-self.cost)

        self.accuracy_summary = tf.scalar_summary("accuracy", self.accuracy)
        self.cost_summary = tf.scalar_summary("cost", self.cost)
    
    def train(self, x, y):
        if( self.objective == 'reconstruction' ):
            y = x
        self.trainingStep.run(feed_dict={self.x: x, self.target: y})
    
    def evaluate(self, x, y):
        if( self.objective == 'reconstruction' ):
            y = x
        return self.accuracy.eval(feed_dict={self.x: x, self.target: y})
    
    def predict(self, x):
        return self.y.eval(feed_dict={self.x: x})

    def getResponseAtLayer(self, x, l):
        assert (l < len(self.layers))
        return self.layers[l].h.eval(feed_dict={self.x: x})

    def encode(self, x):
        return self.encoded.eval(feed_dict={self.x: x})

    def decode(self, x):
        encoded = tf.placeholder(tf.float32, [1] + [k for k in self.encoded.get_shape()[1:]])
        #encoded = tf.placeholder(tf.float32, [1, self.encoded.get_shape()[1], self.encoded.get_shape()[2], self.encoded.get_shape()[3]])

        curx = encoded
        for i in range(len(self.layers)-1):
            l = self.layers[len(self.layers)-i-1]
            decodingLayer = l.invertedBy(layer=l, batch_size=1)
            decodingLayer.link(curx)
            curx = decodingLayer.h

        return curx.eval(feed_dict={encoded: x})

    def getVariables(self):
        variables = []
        for l in self.layers:
            variables += l.trainables
        return variables