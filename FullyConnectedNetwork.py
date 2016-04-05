import tensorflow as tf
from NetworkLayer import FullyConnectedLayer, FlatInputLayer
from WeightInit import WeightInit

FullyConnectedNetworkDefinition = {
	'inputLayer' : FlatInputLayer(inputsize=784),
	'hiddenLayers' : [
	    FullyConnectedLayer(inputsize=784,outputsize=1000,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh),
	    FullyConnectedLayer(inputsize=1000,outputsize=200,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh)
	    ],
	'outputLayer' : FullyConnectedLayer(inputsize=200,outputsize=10,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.softmax)
}