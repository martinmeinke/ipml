
import os
import logging
import numpy as np

from Utility import LoadPickleFile, SavePickleFile

class FeatureProvider(object):
    """
    Provides access to features by loading pickled files instead of using a feature extractor and a data source.
    There will be:
    TrainData, TrainLabels, ValidationData, ValidationLabels, TestData, TestLabels
    Each *Data member will be an m*n matrix with m data points and n features.
    The *Labels member will be an m*1 matrix with the labels corresponding to the m data points
    """
    SAVEPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../saved/extracted_features")

    def __init__(self, dataprovider, FeatureExtractor):
        self._dataprovider = dataprovider
        self._featureExtrator = FeatureExtractor
        self.TrainData = None
        self.TrainLabels = None
        self.ValidationData = None
        self.ValidationLabels = None
        self.TestData = None
        self.TestLabels = None

    def initialize(self):
        logging.info("Loading FeatureProvider. Intializing FeatureExtractor")
        self._featureExtrator.initialize(self._dataprovider.TrainData)

        logging.info("Extracting features from TrainData")
        self.TrainData = np.mat(self._featureExtrator.extract(self._dataprovider.TrainData))
        self.TrainLabels = np.mat(self._dataprovider.TrainLabels).transpose()
        
        logging.info("Extracting features from ValidationData")
        self.ValidationData = np.mat(self._featureExtrator.extract(self._dataprovider.ValidationData))
        self.ValidationLabels = np.mat(self._dataprovider.ValidationLabels).transpose()

        logging.info("Extracting features from TestData")
        self.TestData = np.mat(self._featureExtrator.extract(self._dataprovider.TestData))
        self.TestLabels = np.mat(self._dataprovider.TestLabels).transpose()
        
    def saveToFile(self, path = ""):
        path = path or self.SAVEPATH
        extractorState = self._featureExtrator.saveState()
        logging.info("Saving extracted features")
        SavePickleFile(self.SAVEPATH, (self.TrainData, self.TrainLabels, self.ValidationData, self.ValidationLabels, self.TestData, self.TestLabels, extractorState))

    def loadFromFile(self, path = ""):
        path = path or self.SAVEPATH
        logging.info("Loading extracted features")
        self.TrainData, self.TrainLabels, self.ValidationData, self.ValidationLabels, self.TestData, self.TestLabels, extractorState = LoadPickleFile(self.SAVEPATH)
        self._featureExtrator.loadState(extractorState)
