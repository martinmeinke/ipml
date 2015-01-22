from rf.rnd_forest import *
import numpy as np
from os import path
from SavedFeatureProvider import SavedFeatureProvider

import unittest

class RFTest(unittest.TestCase):
    DATA_DIR = path.join(path.dirname(__file__), 'testdata')
    CATFILE = path.join(DATA_DIR, "cat_vectors")
    DOGFILE = path.join(DATA_DIR, "dog_vectors")
    EXTRACTORFILE = path.join(DATA_DIR, "texel_features")


    @classmethod
    def setUpClass(cls):
        logging.basicConfig(filename="test_rf_firstfeatures.log")

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.sfp = SavedFeatureProvider(self.CATFILE, self.DOGFILE, self.EXTRACTORFILE)
        self.sfp.load()

    def testRF(self):     
        #Training 
        
        forest = RandomForest(self.sfp.TrainData.tolist(), self.sfp.TrainLabels.tolist())
        #cProfile.runctx('forest.generate_forest()',globals(),locals())
        #forest.parallel_generate_forest()
        forest.generate_forest()
        predictions = forest.predict(self.sfp.TrainData.tolist())
        acc = self.getAccuracy(predictions, self.sfp.TrainLabels.tolist())
        logging.info("Error rate on train set: " + str(acc * 100) + "%")
        print acc
    
        #Validation
        predictions = forest.predict(self.sfp.ValidationData.tolist())
        acc = self.getAccuracy(predictions, self.sfp.ValidationLabels.tolist())
        logging.info("Error rate on validation set: " + str(acc * 100) + "%")
        print acc
    
    def getAccuracy(self, list1, list2):
        """
        Returns the error rate between list1 and list2
        """
        size = len(list1)
        count = 0
        for i in xrange(size):
            if list1[i] != list2[i][0]:
                count += 1
        return count / float(size)

if __name__ == "__main__":
    unittest.main()