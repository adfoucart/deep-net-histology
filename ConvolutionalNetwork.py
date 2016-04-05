import tensorflow as tf
from NetworkLayer import FullyConnectedLayer, FlatInputLayer, ConvolutionLayer, VectorToImageLayer, ImageToVectorLayer, MaxPoolingLayer
from WeightInit import WeightInit

ConvolutionalNetworkDefinition = {
	'inputLayer' : FlatInputLayer(inputsize=784),
	'hiddenLayers' : [
		VectorToImageLayer(imagesize=[28,28,1]),
	    ConvolutionLayer(kernelsize=5,channels=1,features=32,stride=1,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.relu),
	    MaxPoolingLayer(poolingSize=2),
	    ConvolutionLayer(kernelsize=5,channels=32,features=64,stride=1,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.relu),
	    MaxPoolingLayer(poolingSize=2),
	    ImageToVectorLayer(imagesize=[7,7,64]),
	    FullyConnectedLayer(inputsize=7*7*64,outputsize=1024,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh)
	    ],
	'outputLayer' : FullyConnectedLayer(inputsize=1024,outputsize=10,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.softmax)
}