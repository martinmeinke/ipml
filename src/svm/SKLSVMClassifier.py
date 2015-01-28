
import logging
from Classifier import Classifier
from sklearn.svm import SVC
from numpy import shape

class SKLSVMClassifier(Classifier):
    '''
    A SVM (support vector machine) classifier that uses the algothms provided by scikit learn (sklearn)
    '''
    Name = "SciKit Learn Support Vector Machine"

    def __init__(self, featureProvider):
        self._fp = featureProvider
        self.TrainingFileName = "SKLearnSVMTraining"
        self.Training = None

    def train(self, C=10, gamma=0.0001):
        logging.info("Use scientific learn!")
        self.Training = SVC(C=C, gamma=gamma)
        self.Training.fit(self._fp.TrainData, self._fp.TrainLabels.A1)
        logging.info("Finished learning")
    
    def testDataSet(self, dataSet, dataLabels):
        dataLabels = dataLabels.A1
        m,n = shape(dataSet)
        logging.info("Use scientific learn!")
        score = self.Training.score(dataSet, dataLabels)
        logging.info("Score is %s", str(score))

        prediction = self.Training.predict(dataSet)
        errorCount = sum((1 if dataLabels[i] != prediction[i] else 0 for i in range(0, m)))
        return float(errorCount) / m 
