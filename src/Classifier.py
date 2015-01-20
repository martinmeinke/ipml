
import os
import logging
from os import path
from Utility import LoadPickleFile, SavePickleFile

class Classifier(object):
    TRAINSAVEDIR = path.join(path.dirname(path.abspath(__file__)), "../trainings/")
    '''
    Base class of the implemented classifiers. To have support for all features,
    do the following when deriving this class:
        - Implement a class variable "Name" that identifies this classifier in a human readable form
        - The constructor gets one additional parameter. Depending on your classifier it's either
          a DataProvider or a FeatureProvider
        - Implement a member variable "TrainingFileName" that suggests a name for the file to save
          the training in
        - Implement a "train" methods that gets custom arguments. It should train your classifier
          from the Data/FeatureProvider's TrainData set.
          Every important trained data should be saved in a custom class that is available as
          the member variable "Training" (this is important, so the training can be saved/loaded)
        - Implement a "testValidationSet" method that takes no arguments and tests the
          Data/FeatureProvider's ValidationData against your trained classifier.
          If you need arguments for this, save them in the "Training".
          This function should return the error rate.
    '''

    def __init__(self):
        self.TrainingFileName = None
        self.Training = None
        raise Exception("Not implemented")

    def train(self):
        raise Exception("Not implemented")

    def testValidationSet(self):
        raise Exception("Not implemented")
    
    def loadTraining(self):
        logging.info("Loading training for %s", self.Name)
        self._checkTrainingFilename();
        self.Training = LoadPickleFile(path.join(self.TRAINSAVEDIR, self.TrainingFileName))

    def saveTraining(self):
        logging.info("Saving training for %s", self.Name)
        self._checkTrainingFilename();
        if not path.exists(self.TRAINSAVEDIR):
            os.mkdir(self.TRAINSAVEDIR)
        SavePickleFile(path.join(self.TRAINSAVEDIR, self.TrainingFileName), self.Training)
    
    def _checkTrainingFilename(self):
        if not self.TrainingFileName:
            msg = "TrainingFileName not set for classifier %s" % self.Name
            logging.error(msg)
            raise Exception(msg)
        
        