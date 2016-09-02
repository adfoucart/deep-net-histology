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

from networkDefinitions import *

# mnistFullyConnectedClassif(False)
#mnistSimpleClassif(True)
#mitos12ConvAutoEncoder3(False, False, 20000)
#mitos12ClassifierFromConvAutoEncoder3(False, False, 50000)
# mitos12ConvAutoEncoder4(True, True, 20000)
mitos12ClassifierFromConvAutoEncoder4(True, True, 20000)
# jphConvAutoEncoder(True, True)
# mitos12AutoEncoderTrainedByLayers(False, 2, True, 0)
# mitos12FullyConnectedAE(False, True, 10000)
# visuNetwork()