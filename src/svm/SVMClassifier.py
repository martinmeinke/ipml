
import logging
from smo import smoP, kernelTrans
from numpy import nonzero, shape, multiply, sign
from Classifier import Classifier

class SVMTraining(object):
    def __init__(self):
        self.B = None
        self.Alphas = None
        self.SupportVectors = None
        self.SVIndices = None
        self.SVLabels = None
        self.UsedKernel = None

class SVMClassifier(Classifier):
    '''
    A SVM (support vector machine) classifier based on the Sequential Minimal Optimization algorithm
    '''
    Name = "Support Vector Machine"

    def __init__(self, featureProvider):
        self._fp = featureProvider
        self.TrainingFileName = "SVMTraining"
        self.Training = None

    def train(self, C=200, toler=0.0001, maxIter=1000, kTup=('rbf', 1.3)):
        logging.info("SVM Parameters: C=%f, toler=%f, maxIter=%d, kTup=%s", C, toler, maxIter, str(kTup))
        logging.info("Shape of training data: %s", str(self._fp.TrainData.shape))
        self.Training = SVMTraining()
        # actual training
        self.Training.B, self.Training.Alphas = smoP(self._fp.TrainData, self._fp.TrainLabels, C, toler, maxIter, kTup);
        
        # save important additional info
        self.Training.UsedKernel = kTup
        svInd = nonzero(self.Training.Alphas.A > 0)[0]
        self.Training.SVIndices = svInd
        self.Training.SupportVectors = self._fp.TrainData[svInd]
        self.Training.SVLabels = self._fp.TrainLabels[svInd];

    def testDataSet(self, dataSet, dataLabels):
        errorCount = 0
        m,n = shape(dataSet)
        for i in range(m):
            kernelEval = kernelTrans(self.Training.SupportVectors, dataSet[i,:], self.Training.UsedKernel)
            predict = kernelEval.T * multiply(self.Training.SVLabels, self.Training.Alphas[self.Training.SVIndices]) + self.Training.B
            if sign(predict) != sign(dataLabels[i]):
                errorCount += 1
        return float(errorCount) / m
