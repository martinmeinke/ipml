'''
Created on Dec 9, 2014

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
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "48_medium"
# 0 = train, 1 = kaggle eval
RUN_MODE = 0
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

    PARTITIONING = MP["PARTITIONING"]
    DATASET = MP["DATASET"]
    KAGGLE_TEST_SET = MP["KAGGLE_TEST_SET"]

    C1 = MP["C1"]
    C2 = MP["C2"]
    C3 = MP["C3"]
    POOL1 = MP["POOL1"]
    POOL2 = MP["POOL2"]
    POOL3 = MP["POOL3"]
    H1OUT = MP["H1OUT"]
    H2OUT = MP["H2OUT"]

    # type of activation function
    POOL1ACTIVATION = MP["POOL1ACTIVATION"]
    POOL2ACTIVATION = MP["POOL2ACTIVATION"]
    POOL3ACTIVATION = MP["POOL3ACTIVATION"]
    H1ACTIVATION = MP["H1ACTIVATION"]
    H2ACTIVATION = MP["H2ACTIVATION"]

    # regularization coefficients
    L1 = MP["L1"]
    L2 = MP["L2"]

    LR_LAMBDA = MP["LR_LAMBDA"]
    LR_DECAY = MP["LR_DECAY"]

    if RUN_MODE == 0:
        BATCH_SIZE = MP["BATCH_SIZE"]
    else:
        BATCH_SIZE = 100  # n % batch_size == 0

    IMAGE_SHARE = MP["IMAGE_SHARE"]
    TRAIN_ENHANCED = MP["TRAIN_ENHANCED"]
    N_EPOCHS = MP["N_EPOCHS"]

    ####################

    def testNetwork(self):
        # create a trainer
        self.trainer = Trainer(self.LR_LAMBDA, self.BATCH_SIZE, self.N_EPOCHS, self.LR_DECAY)

        if RUN_MODE == 0:
            self.trainer.prepare_training_data(DATASET, PARTITIONING, self.IMAGE_SHARE, self.TRAIN_ENHANCED)
        elif RUN_MODE == 1:
            self.trainer.prepare_kaggle_test_data(KAGGLE_TEST_SET)
            # make sure every test item is being looked at
            assert self.trainer.n_test_samples % self.trainer.batch_size == 0

        input_size = self.trainer.get_input_size()

        # build network structure
        self.network = Network()
        self.network.add_layer(ConvLayer(self.network.rng, self.C1, input_size, self.trainer.batch_size))
        self.network.add_layer(SubsamplingLayer(self.network.rng, self.POOL1, activation=self.POOL1ACTIVATION))
        self.network.add_layer(ConvLayer(self.network.rng, self.C2, batch_size=self.trainer.batch_size))
        self.network.add_layer(SubsamplingLayer(self.network.rng, self.POOL2, activation=self.POOL2ACTIVATION))
        self.network.add_layer(ConvLayer(self.network.rng, self.C3, batch_size=self.trainer.batch_size))
        self.network.add_layer(SubsamplingLayer(self.network.rng, self.POOL3, activation=self.POOL3ACTIVATION))
        self.network.add_layer(HiddenLayer(self.network.rng, self.H1OUT, activation=self.H1ACTIVATION))
        self.network.add_layer(HiddenLayer(self.network.rng, self.H2OUT, activation=self.H2ACTIVATION))
        self.network.add_layer(SoftMax(self.network.rng, 2, self.L1, self.L2))

        # set the network input
        net_input = self.trainer.x.reshape(self.trainer.get_input_size())
        self.network.set_input(net_input)

        # after we know the input, we can build the structure
        self.network.fix_structure()

        # trainer needs to know the network
        self.trainer.network = self.network
        self.trainer.prepare_models(RUN_MODE)

        # validate after each epoch
        self.trainer.validation_frequency = self.trainer.n_train_batches

        dataset_manager = DatasetManager("../../models")

        if dataset_manager.dataset_available(MODEL_NAME):
            logger.info("Loading saved model: {}".format(MODEL_NAME))
            self.network.model_params = []
            for x in dataset_manager.load(MODEL_NAME):
                self.network.model_params.append(x)
            self.network.restore_params()

        # Train network or evaluate existing one?
        if RUN_MODE == 0:
            self.trainer.run_training()
            # store the trained model
            dataset_manager.store(self.network.model_params, MODEL_NAME)
        elif RUN_MODE == 1:
            labels = self.trainer.evaluate_test_set()

            # write file which can be uploaded to kaggle for eval
            with open("../../kaggle_out.csv", "w") as f:
                f.write("id,label\n")
                for num, label in zip(range(1, len(labels)+1), labels):
                    f.write("{},{}\n".format(num, label))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()