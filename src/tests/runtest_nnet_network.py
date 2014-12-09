'''
Created on Dec 9, 2014

@author: martin
'''
import unittest
from nnet.Network import Network
from nnet.ConvolutionalLayer import ConvolutionalLayer
from nnet.HiddenLayer import HiddenLayer
from classifiers.LogisticRegression import LogisticRegression
from nnet.SubsamplingLayer import SubsamplingLayer

class Test(unittest.TestCase):

    def setUp(self):
        self.network = Network()
#         self.network.add_layer(ConvolutionalLayer(self.network.rng, filter_shape, image_shape))
#         self.network.add_layer(SubsamplingLayer(poolsize))
#         self.network.add_layer(ConvolutionalLayer(self.network.rng, filter_shape, image_shape))
#         self.network.add_layer(SubsamplingLayer(poolsize))
#         self.network.add_layer(HiddenLayer(self.network.rng, n_in, n_out, W, b, activation))
#         self.network.add_layer(HiddenLayer(self.network.rng, n_in, n_out, W, b, activation))
#         self.network.add_layer(LogisticRegression())
        pass


    def tearDown(self):
        pass


    def testName(self):
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()