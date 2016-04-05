import tensorflow as tf
from MNISTData import MNISTData 
from MITOS12Data import MITOS12Data
from JPHAlguesData import JPHAlguesData
from matplotlib import pyplot as plt
import numpy as np
from sys import stdout
from networkViz import *
from NetworkLayer import *
from WeightInit import WeightInit

from Network import Network

from networkDefinitions import mnistFullyConnectedClassif, mitos12ConvAutoEncoder3, jphConvAutoEncoder, mitos12AutoEncoderTrainedByLayers, mitos12FullyConnectedAE, mitos12ClassifierFromConvAutoEncoder3

# mnistFullyConnectedClassif(False)
mitos12ConvAutoEncoder3(False, False, 10000)
# mitos12ClassifierFromConvAutoEncoder3(True, True, 50000)
# jphConvAutoEncoder(True, True)
# mitos12AutoEncoderTrainedByLayers(False, 2, True, 0)
# mitos12FullyConnectedAE(False, True, 10000)
# visuNetwork()