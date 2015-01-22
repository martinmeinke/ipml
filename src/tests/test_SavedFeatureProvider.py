'''
Created on Jan 20, 2015

@author: stefan
'''
import unittest
import logging
from os import path
from SavedFeatureProvider import SavedFeatureProvider 
from LoggingSetup import LoggingSetup



class Test(unittest.TestCase):
    DATA_DIR = path.join(path.dirname(__file__), 'testdata')
    CATFILE = path.join(DATA_DIR, "cat_vectors")
    DOGFILE = path.join(DATA_DIR, "dog_vectors")
    EXTRACTORFILE = path.join(DATA_DIR, "texel_features")
    
    NUM_FEATURES=100
    NUM_VALIDATION=30
    NUM_TRAIN=170
    
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(filename="testSavedFeatureProvider.log")

    def testLoad(self):     
        sfp = SavedFeatureProvider(self.CATFILE, self.DOGFILE, self.EXTRACTORFILE)
        sfp.load()
        self.assertEqual(sfp.TrainData.shape, (self.NUM_TRAIN, self.NUM_FEATURES), "Train data matrix has wrong size")
        self.assertEqual(sfp.ValidationData.shape, (self.NUM_VALIDATION, self.NUM_FEATURES), "Validation data matrix has wrong size")
        self.assertEqual(sfp.TrainLabels.shape, (self.NUM_TRAIN, 1), "Train label matrix has wrong size")
        self.assertEqual(sfp.ValidationLabels.shape, (self.NUM_VALIDATION, 1), "Validation label matrix has wrong size")
        

if __name__ == "__main__":
    unittest.main()