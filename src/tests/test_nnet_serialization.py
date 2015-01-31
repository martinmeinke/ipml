'''
Created on Jan 27, 2015

@author: martin
'''
import unittest
from nnet.Network import Network
from nnet.Trainer import Trainer
from nnet.ConvLayer import ConvLayer
from nnet.HiddenLayer import HiddenLayer
from nnet.LogisticRegression import SoftMax
from nnet.SubsamplingLayer import SubsamplingLayer
from imageio.dataset_manager import DatasetManager
import theano.tensor as T
import logging
import json
import numpy

class Test(unittest.TestCase):
    """Tests the model serialization
    """

    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testName(self):
        batch_size = 128;
        input_size = (batch_size, 1, 96, 96)
        # build network structure
        network = Network()
        network.add_layer(ConvLayer(network.rng, [10, 1, 5, 5], input_size, batch_size, regularizer_weight=1))
        network.add_layer(SubsamplingLayer(network.rng, [2, 2], activation=0))
        network.add_layer(HiddenLayer(network.rng, 300, activation=0, regularizer_weight=1))
        network.add_layer(SoftMax(network.rng, 2, 0, 0, regularizer_weight=1))
        
        # set the network input
        x = T.matrix('x')  # the data is presented as rasterized images
        net_input = x.reshape(input_size)
        network.set_input(net_input)
        
        # after we know the input, we can build the structure
        network.fix_structure()
        
        random_weights = [l.params for l in network.layers]
        par_shapes = []
        weight_values = []
        for lp in random_weights:
            for p in lp:
                par_shapes.append(p.get_value().shape)
                weight_values.append(p.get_value())
            
        dm = DatasetManager("../../serializer_test")
        dm.store(network.model_params, "test_model")
        
        weights_intermediate = []
        # randomize weights
        i = 0
        for lp in random_weights:
            for p in lp:
                sum = reduce(lambda x,y: x*y, par_shapes[i])
                newvals = numpy.random.randn(sum)
                newvals = newvals.reshape(par_shapes[i])
                newvals = numpy.asarray(newvals, dtype=numpy.float32)
                p.set_value(newvals)
                weights_intermediate.append(newvals)
                i+=1
        
        network.restore_params(dm.load("test_model"))
            
        random_weights = [l.params for l in network.layers]
        par_shapes = []
        weight_values2 = []
        for lp in random_weights:
            for p in lp:
                par_shapes.append(p.get_value().shape)
                weight_values2.append(p.get_value())
                
        for e1, e2 in zip(weight_values, weight_values2):
            self.assertTrue((e1==e2).all(), "Deserialized model different from start")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()