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
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "real_v2"
PARTITIONING = (85, 10, 5)
DATASET = "../../data/train_images_96_100percent"
# DATASET = "../../data/train_images_128_100percent"


def readParams(f):
    with open(f) as json_file:
        json_data = json.load(json_file)
        return json_data


class Test(unittest.TestCase):
    # we need a network and trainer...
    network = None
    trainer = None

    PARAMETERS = readParams("../../modelparams.json")
    MP = PARAMETERS[MODEL_NAME]

    # HYPERPARAMETERS
#     C1 = (50, 1, 9, 9)
#     C2 = (80, 50, 5, 5)
#     C3 = (100, 80, 3, 3)
#     POOL1 = (2, 2)
#     POOL2 = (2, 2)
#     POOL3 = (2, 2)
#     H1OUT = 900
#     H2OUT = 1200
# 
#     POOL1ACTIVATION = TANH
#     POOL2ACTIVATION = TANH
#     POOL3ACTIVATION = TANH
#     H1ACTIVATION = RELU
#     H2ACTIVATION = RELU
# 
#     LR_LAMBDA = 0.04
#     LR_DECAY = True
#     BATCH_SIZE = 128
#     N_EPOCHS = 100

    C1 = MP["C1"]
    C2 = MP["C2"]
    C3 = MP["C3"]
    POOL1 = MP["POOL1"]
    POOL2 = MP["POOL2"]
    POOL3 = MP["POOL3"]
    H1OUT = MP["H1OUT"]
    H2OUT = MP["H2OUT"]

    POOL1ACTIVATION = MP["POOL1ACTIVATION"]
    POOL2ACTIVATION = MP["POOL2ACTIVATION"]
    POOL3ACTIVATION = MP["POOL3ACTIVATION"]
    H1ACTIVATION = MP["H1ACTIVATION"]
    H2ACTIVATION = MP["H2ACTIVATION"]

    LR_LAMBDA = MP["LR_LAMBDA"]
    LR_DECAY = MP["LR_DECAY"]
    BATCH_SIZE = MP["BATCH_SIZE"]
    N_EPOCHS = MP["N_EPOCHS"]

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