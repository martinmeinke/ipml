from numpy import *
from svm.smo import *
from os import path

import unittest

class SVMSMPOTest(unittest.TestCase):
    trainDataFile = path.join(path.dirname(__file__), 'testdata/trainSetRBF.txt')
    testDataFile = path.join(path.dirname(__file__), 'testdata/testSetRBF.txt')


    def setUp(self):
        self.trainData, self.trainLabels = self.loadDataSet(self.trainDataFile)

    def loadDataSet(self, fileName):
        dataMat = []; labelMat = []
        fr = open(fileName)
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
        return dataMat,labelMat

    def testRbf(self, k1=1.3):
        # actually train
        b,alphas = smoP(self.trainData, self.trainLabels, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
        datMat=mat(self.trainData);
        labelMat = mat(self.trainLabels).transpose()

        svInd=nonzero(alphas.A>0)[0]
        sVs=datMat[svInd] #get matrix of only support vectors
        labelSV = labelMat[svInd];
        numSVs = shape(sVs)[0]
        self.assertGreater(numSVs, 15,"Less than 20 support vectors: {}".format(numSVs))
        self.assertLess(numSVs, 30,"More than 30 support vectors: {}".format(numSVs))

        # Check on train data
        m,n = shape(datMat)
        errorCount = 0
        for i in range(m):
            kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
            predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
            if sign(predict)!=sign(self.trainLabels[i]): errorCount += 1
        self.assertLess(float(errorCount)/m, 0.15, "Training error rate too high")

        # Apply to test data
        testData, testLabels = self.loadDataSet(self.testDataFile)
        errorCount = 0
        datMat=mat(testData); labelMat = mat(testLabels).transpose()
        m,n = shape(datMat)
        for i in range(m):
            kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
            predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
            if sign(predict)!=sign(testLabels[i]): errorCount += 1
        self.assertLess(float(errorCount)/m, 0.25, "Test error rate too high")

if __name__ == "__main__":
    unittest.main()