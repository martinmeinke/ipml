'''
Created on Dec 9, 2014

@author: martin
'''
import unittest
from nnet.Network import Network, Trainer

from nnet.ConvolutionalLayer import ConvolutionalLayer
from nnet.HiddenLayer import HiddenLayer
from nnet.LogisticRegression import SoftMax
from nnet.SubsamplingLayer import SubsamplingLayer
import theano.tensor as T
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATASET = "../../data/train_images_10percent"
PARTITIONING = (65,17,18)

class Test(unittest.TestCase):
    
    network = None
    trainer = None
    
    # number of kernels, number of feature maps in prev. layer, kernel shape
    C1 = (10, 1, 5, 5)
    C2 = (12, 10, 7, 7)
    C3 = (14, 12, 5, 5)
    POOL1 = (2, 2)
    POOL2 = (2, 2)
    POOL3 = (4, 4)
    H1OUT = 300
    H2OUT = 300
    H1ACTIVATION = T.tanh
    H2ACTIVATION = T.tanh
    
    def setUp(self):
        #create a trainer
        self.trainer = Trainer()
        self.trainer.prepare_data(DATASET, PARTITIONING)
        
        #build network structure
        self.network = Network()
        self.network.add_layer(ConvolutionalLayer(self.network.rng, self.C1, self.trainer.get_input_size(), self.trainer.batch_size))
        self.network.add_layer(SubsamplingLayer(self.network.rng, self.POOL1))
        self.network.add_layer(ConvolutionalLayer(self.network.rng, self.C2, batch_size=self.trainer.batch_size))
        self.network.add_layer(SubsamplingLayer(self.network.rng, self.POOL2))
        self.network.add_layer(ConvolutionalLayer(self.network.rng, self.C3, batch_size=self.trainer.batch_size))
        self.network.add_layer(SubsamplingLayer(self.network.rng, self.POOL3))
        self.network.add_layer(HiddenLayer(self.network.rng, self.H1OUT, activation=self.H1ACTIVATION))
        self.network.add_layer(HiddenLayer(self.network.rng, self.H2OUT, activation=self.H2ACTIVATION))
        self.network.add_layer(SoftMax(self.network.rng, 2))
        
        #set the network input
        net_input = self.trainer.x.reshape(self.trainer.get_input_size())
        self.network.set_input(net_input)
        #after we know the input, we can build the structure
        self.network.fix_structure()
        
        #trainer needs to know the network
        self.trainer.network = self.network
        self.trainer.prepare_models()
        self.trainer.run_training()
        pass


    def tearDown(self):
        pass


    def testName(self):
        pass

    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    