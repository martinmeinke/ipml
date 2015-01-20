import os
import pickle
import logging

import numpy as np
from os import path

class SavedFeatureProvider(object):
    """
    Provides access to features by loading pickled files instead of using a feature extractor and a data source.
    There will be:
    TrainData, TrainLabels, ValidationData, ValidationLabels, TestData, TestLabels
    Each *Data member will be an m*n matrix with m data points and n features.
    The *Labels member will be an m*1 matrix with the labels corresponding to the m data points
    """

    DATADIR = path.join(path.dirname(path.abspath(__file__)), "../features/")
    CAT_FEATURES_FILE = path.join(DATADIR, "cat_vectors")
    DOG_FEATURES_FILE = path.join(DATADIR, "dog_vectors")
    EXTRACTOR_DATA_FILE = path.join(DATADIR, "texel_features")
    VALIDATION_SET_SIZE = 15 # 15 percent of the labeled data is validation data and not training data
    CAT_LABEL = 1
    DOG_LABEL = -1

    def __init__(self, catDataFile = "", dogDataFile = "", extractorDataFile = ""):
        self._catfile = catDataFile or self.CAT_FEATURES_FILE
        self._dogfile = dogDataFile or self.DOG_FEATURES_FILE
        self._extractorfile = extractorDataFile or self.EXTRACTOR_DATA_FILE

    def load(self):
        """
        Loads the data from pickle files and initializes the important members
        """
        logging.info("Trying to load cat features")
        catfeatures = np.mat(self._loadPickleFile(self._catfile))
        logging.info("Trying to load dog features")
        dogfeatures = np.mat(self._loadPickleFile(self._dogfile))

        # get number of features and check for validity
        self.NumFeatures = catfeatures.shape[1]
        if dogfeatures.shape[1] != self.NumFeatures:
            self._logAndRaise("Dog data has %d features instead of %d" % (dogfeatures.shape[1], self.NumFeatures))

        # Let's define TrainingData, TrainingLabels, ValidationData, ValidationLabels
        self._arrangeDataSets(catfeatures, dogfeatures, self.VALIDATION_SET_SIZE)
        # no support for TestData, TestLabels yet
        self.TestData = None
        self.TestLabels = None

        logging.info("Trying to load extractor data")
        self.ExtractorData = self._loadPickleFile(self._extractorfile)

    def _arrangeDataSets(self, catfeatures, dogfeatures, percentValidation):
        """
        Takes two matrices, one with cat features and one with dog features.
        It will take random <percentValidation>% of each data set to create the randomly mixed validation set.
        The rest will be a randomly mixed training set. The corresponding members will be initialized
        """
        # first split cat and dog data in training and validation set
        logging.info("Split and label the data into train and validation sets. (%i*%i cat features, %i*%i dog features before)" %
                     (catfeatures.shape + dogfeatures.shape))
        catTrain, catVal = self._splitAndLabel(catfeatures, percentValidation, self.CAT_LABEL)
        dogTrain, dogVal = self._splitAndLabel(dogfeatures, percentValidation, self.DOG_LABEL)
        logging.info("After split and label: %i*%i Cat Train, %i*%i Cat Validation. %i*%i Dog Train, %i*%i Dog Validation" %
                     (catTrain.shape + catVal.shape + dogTrain.shape + dogVal.shape))

        # now concat train and test data, split to actual data and labels
        logging.info("Concat, Mix and Split data")
        self.TrainData, self.TrainLabels = self._joinAndMixAndUnlabel(catTrain, dogTrain)
        self.ValidationData, self.ValidationLabels = self._joinAndMixAndUnlabel(catVal, dogVal)
        logging.info("Train Data Size: %i*%i, Train Label Size: %i*%i, Validation Data Size: %i*%i, Validation Label Size: %i*%i" %
                     (self.TrainData.shape + self.TrainLabels.shape + self.ValidationData.shape + self.ValidationLabels.shape))


    def _splitAndLabel(self, features, percentValidation, label):
        """
        Splits the feature matrix <features> into train an test data. Both resulting matrices
        will have an additional last column with the label in it. The label is defined by <label>
        """
        nData = features.shape[0]
        nValData = int(nData * (float(percentValidation) / 100))
        logging.info("Taking %i from %i data points for validation set" % (nValData, nData))
        # pick columns randomly for validation data and create labeled validation data matrix
        valRows = np.random.choice(nData, nValData, replace=False)
        valLabels = np.ones((nValData, 1)) * label
        valData = np.append(features[valRows, :], valLabels, axis=1) # append labels as last column
        # now take the rest to create labeled train data matrix
        trainLabels = np.ones((nData - nValData, 1)) * label
        trainData = np.append(np.delete(features, valRows, axis=0), trainLabels, axis=1)
        return trainData, valData

    def _joinAndMixAndUnlabel(self, catfeatures, dogfeatures):
        """
        Joins two labeled matrices with the same number of features (the label being the last column),
        then mixes the joined matrix and split it by data/label. The results are 2 matrices:
        1. A data matrix with size (m1+m2)*n; m1 being the number of cat data, m2 the number of dog data, and n the number of features
        2. A label matrix with size (m1+m2)*1; each row with the label corresponding to the data rows in the first result  
        """
        joined = np.append(catfeatures, dogfeatures, axis=0) # join it
        np.random.shuffle(joined) # mix it
        return np.mat(joined[:,:-1]), np.mat(joined[:,-1]) # split data and labels


    def _loadPickleFile(self, path):
        """
        Simply load data from a pickle file
        """
        logging.info("Attempt to load data to file '%s'", path)
        if os.path.exists(path):
            with open(path,'rb') as f:
                data = pickle.load(f)
                logging.info("Data loaded sucessfully")
                return data
        else:
            self._logAndRaise("File '%s' does not exist" % path)

    def _logAndRaise(self, msg):
        logging.error(msg)
        raise Exception(msg)
