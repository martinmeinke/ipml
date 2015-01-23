from rnd_forest import RandomForest, ForestParams
from Classifier import Classifier
    
    
class RFClassifier(Classifier):
    '''
    A random forest classifier 
    '''
    Name = "Random Forest"
    params = None

    def __init__(self, featureProvider):
        self._fp = featureProvider
        self.TrainingFileName = "RFTraining"
        self.Training = None
        self.RF = None
        
    def train(self):
        self.RF = RandomForest(self._fp.TrainData.tolist(), self._fp.TrainLabels.tolist())
        self.RF.parallel_generate_forest()
        
    def testValidationSet(self):
        predictions = self.RF.predict(self._fp.ValidationData.tolist())
        acc = self.getAccuracy(predictions, self._fp.ValidationLabels.tolist())
        return acc
    
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