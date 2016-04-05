import tensorflow as tf
from NetworkLayer import *
from WeightInit import WeightInit

AutoEncoderNetworkDefinition = {
	'inputLayer' : FlatInputLayer(inputsize=784),
	'hiddenLayers' : [
	    FullyConnectedLayer(inputsize=784,outputsize=1000,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh),
	    FullyConnectedLayer(inputsize=1000,outputsize=200,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh),
    ]
}

Mitos12AutoEncoderNetworkDefinition = {
    'inputLayer' : FlatInputLayer(inputsize=64*64*3),
    'hiddenLayers' : [
        FullyConnectedLayer(inputsize=64*64*3,outputsize=1000,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh),
        FullyConnectedLayer(inputsize=1000,outputsize=200,weightInitFunc=WeightInit.truncatedNormal,biasInitFunc=WeightInit.positive,activationFunc=tf.nn.tanh),
        ]
}

Mitos12ConvAutoEncoderNetworkDefinition = {
    'inputLayer' : ImageInputLayer(width=64,height=64,channels=3),
    'hiddenLayers' : [
        ConvolutionLayer(kernelsize=5, channels=3, features=16, stride=2, weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[64,64,3]),
        ConvolutionLayer(kernelsize=5, channels=16, features=64, stride=2,weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[32,32,16])
    ]
}

Mitos12ConvAutoEncoder3NetworkDefinition = {
    'inputLayer' : ImageInputLayer(width=64,height=64,channels=3),
    'hiddenLayers' : [
        ConvolutionLayer(kernelsize=5, channels=3, features=12, stride=2, weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[64,64,3]),
        ConvolutionLayer(kernelsize=5, channels=12, features=48, stride=2,weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[32,32,12]),
        ConvolutionLayer(kernelsize=5, channels=48, features=192, stride=2,weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[16,16,48])
    ]
}

JPHAlguesConvAutoEncoderNetworkDefinition = {
    'inputLayer' : ImageInputLayer(width=128, height=128, channels=3),
    'hiddenLayers': [
        ConvolutionLayer(kernelsize=5, channels=3, features=64, stride=2, weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[128,128,3]),
        ConvolutionLayer(kernelsize=5, channels=64, features=128, stride=2, weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[64,64,64]),
        ConvolutionLayer(kernelsize=5, channels=128, features=256, stride=2, weightInitFunc=WeightInit.truncatedNormal, biasInitFunc=WeightInit.positive, activationFunc=tf.nn.relu, inputshape=[32,32,128])
    ]
}