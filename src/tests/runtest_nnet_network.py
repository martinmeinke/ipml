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
from imageio.dataset_manager import DatasetManager
import theano.tensor as T
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "real_v3"
PARTITIONING = (80,10,10)
DATASET = "../../data/train_images_96_100percent"
# DATASET = "../../data/train_images_128_100percent"

class Test(unittest.TestCase):
    # we need a network and trainer...
    network = None
    trainer = None

    # HYPERPARAMETERS
    C1 = (4, 1, 9, 9)
    C2 = (6, 4, 5, 5)
    C3 = (9, 6, 3, 3)
    POOL1 = (2, 2)
    POOL2 = (2, 2)
    POOL3 = (2, 2)
    H1OUT = 90
    H2OUT = 120

    TANH = 0
    RELU = 1

    POOL1ACTIVATION = TANH
    POOL2ACTIVATION = TANH
    POOL3ACTIVATION = TANH
    H1ACTIVATION = RELU
    H2ACTIVATION = RELU

    LR_LAMBDA = 0.04
    LR_DECAY = True
    BATCH_SIZE = 128
    N_EPOCHS = 100
    ####################

    def testNetwork(self):
        # create a trainer
        self.trainer = Trainer(self.LR_LAMBDA, self.BATCH_SIZE, self.N_EPOCHS, self.LR_DECAY)
        self.trainer.prepare_data(DATASET, PARTITIONING, 100)

        # build network structure
        self.network = Network()
        self.network.add_layer(ConvolutionalLayer(self.network.rng, self.C1, self.trainer.get_input_size(), self.trainer.batch_size))
        self.network.add_layer(SubsamplingLayer(self.network.rng, self.POOL1, activation=self.H1ACTIVATION))
        self.network.add_layer(ConvolutionalLayer(self.network.rng, self.C2, batch_size=self.trainer.batch_size))
        self.network.add_layer(SubsamplingLayer(self.network.rng, self.POOL2, activation=self.H1ACTIVATION))
        self.network.add_layer(ConvolutionalLayer(self.network.rng, self.C3, batch_size=self.trainer.batch_size))
        self.network.add_layer(SubsamplingLayer(self.network.rng, self.POOL3, activation=self.H1ACTIVATION))
        self.network.add_layer(HiddenLayer(self.network.rng, self.H1OUT, activation=self.H1ACTIVATION))
        self.network.add_layer(HiddenLayer(self.network.rng, self.H2OUT, activation=self.H2ACTIVATION))
        self.network.add_layer(SoftMax(self.network.rng, 2))

        # set the network input
        net_input = self.trainer.x.reshape(self.trainer.get_input_size())
        self.network.set_input(net_input)

        # after we know the input, we can build the structure
        self.network.fix_structure()

        # trainer needs to know the network
        self.trainer.network = self.network
        self.trainer.prepare_models()

        # validate after each epoch
        self.trainer.validation_frequency = self.trainer.n_train_batches

        dataset_manager = DatasetManager("../../models")

        if dataset_manager.dataset_available(MODEL_NAME):
            logger.info("Loading saved model: {}".format(MODEL_NAME))
            self.network.model_params = []
            for x in dataset_manager.load(MODEL_NAME):
                self.network.model_params.append(x)
            self.network.restore_params()

        # how often to display validation output
        # visitor patternself.trainer.validation_frequency = self.trainer.n_train_batches
        self.trainer.run_training()

        dataset_manager.store(self.network.model_params, MODEL_NAME)
        pass


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()