from rnd_forest import RandomForest, ForestParams
from FeatureExtraction.FeatureSelection import FeatureSelector
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
        
    def train(self, num_attr, max_tries, subset, min_gain, thres_steps, forest_size, max_depth, select_kbest=False):
        parms = ForestParams(num_attr, max_tries, subset, min_gain, thres_steps, forest_size, max_depth)
        self.Training = RandomForest(self._fp.TrainData, self._fp.TrainLabels, parms)
        if(select_kbest):
            selector = FeatureSelector(self._fp.TrainData, self._fp.TrainLabels)
            kbest = selector.rf_get_k_best(700)
            self.Training.set_kbest(kbest)
            
        self.Training.parallel_generate_forest()
        
    def testDataSet(self, dataSet, dataLabels):
        predictions = self.Training.predict(dataSet.tolist())
        acc = self.getAccuracy(predictions, dataLabels.tolist())
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