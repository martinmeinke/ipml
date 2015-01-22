from numpy import *
from svm.smo import *
from os import path
from SavedFeatureProvider import SavedFeatureProvider

import unittest

class SVMSMPOTest(unittest.TestCase):
    DATA_DIR = path.join(path.dirname(__file__), 'testdata')
    CATFILE = path.join(DATA_DIR, "cat_vectors")
    DOGFILE = path.join(DATA_DIR, "dog_vectors")
    EXTRACTORFILE = path.join(DATA_DIR, "texel_features")


    @classmethod
    def setUpClass(cls):
        logging.basicConfig(filename="test_svm_firstfeatures.log")

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.sfp = SavedFeatureProvider(self.CATFILE, self.DOGFILE, self.EXTRACTORFILE)
        self.sfp.load()

    def testRbf(self, k1=1.3):
        # actually train
        b,alphas = smoP(self.sfp.TrainData, self.sfp.TrainLabels, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important

        svInd=nonzero(alphas.A>0)[0]
        sVs=self.sfp.TrainData[svInd] #get matrix of only support vectors
        labelSV = self.sfp.TrainLabels[svInd];
        numSVs = shape(sVs)[0]

        # Check on train data
        m,n = shape(self.sfp.TrainData)
        errorCount = 0
        for i in range(m):
            kernelEval = kernelTrans(sVs,self.sfp.TrainData[i,:],('rbf', k1))
            predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
            if sign(predict)!=sign(self.sfp.TrainLabels[i]): errorCount += 1
        self.assertLess(float(errorCount)/m, 0.5, "Training error rate too high: {}".format(float(errorCount)/m))

        # Apply to test data
        errorCount = 0
        m,n = shape(self.sfp.ValidationData)
        for i in range(m):
            kernelEval = kernelTrans(sVs,self.sfp.ValidationData[i,:],('rbf', k1))
            predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
            if sign(predict)!=sign(self.sfp.ValidationLabels[i]): errorCount += 1
        self.assertLess(float(errorCount)/m, 0.6, "Validation error rate too high: {}".format(float(errorCount)/m))

if __name__ == "__main__":
    unittest.main()