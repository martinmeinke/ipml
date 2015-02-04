
import logging
from smo import smoP
from numpy import nonzero
from Classifier import Classifier
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theanoSMO import kernelTrans_, toTheanoBool

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
    
    def testDataSet_inner_(self, svs, A, gamma, svLabels, svAlphas, b, realLabel):
        kEvaled = kernelTrans_(svs, A, gamma)
        predict = T.dot(kEvaled.T, (svLabels * svAlphas)) + b
        haveSameSign = toTheanoBool(T.eq(T.sgn(predict), T.sgn(realLabel)))
        return ifelse(haveSameSign, 0, 1)

    def testDataSet(self, dataSet_, dataLabels_):
        dataSet = T.matrix("dataSet")
        labels = T.col("labels")
        svLabels = T.col("svLabels")
        gamma = T.dscalar("gamma")
        svs = T.matrix("supportVectors")
        svAlphas = T.matrix("svAlphas")
        b = T.dscalar("b")
              
        # we need to transpose the result because the results of the per-row actions are usually columns
        errorVec = theano.scan(lambda row, realLabel : self.testDataSet_inner_(svs, row, gamma, svLabels, svAlphas, b, realLabel), sequences=[dataSet, labels])[0]
        errors = T.sum(errorVec)
        
        inputs = [dataSet, labels, svs, svLabels, gamma, svAlphas, b]
        compErrors = theano.function(inputs=inputs, outputs=errors, on_unused_input='ignore')
        
        gamma_ = 1/(-1*self.Training.UsedKernel[1]**2)
        numErrors = compErrors(dataSet_, dataLabels_, self.Training.SupportVectors, self.Training.SVLabels, gamma_, self.Training.Alphas[self.Training.SVIndices], self.Training.B.item(0))
        return float(numErrors) / float(dataSet_.shape[0])
