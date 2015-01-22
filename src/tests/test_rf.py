from rf.rnd_forest import RandomForest
from os import path

import unittest

class RFTest(unittest.TestCase):
    trainFeaturesFile = path.join(path.dirname(__file__), 'testdata/trainFeatures.csv')
    trainLabelsFile = path.join(path.dirname(__file__), 'testdata/trainLabels.csv')
    
    valFeatuesFile = path.join(path.dirname(__file__), 'testdata/valFeatures.csv')
    valLabelsFile = path.join(path.dirname(__file__), 'testdata/valLabels.csv')


    def setUp(self):
        self.trainFeatures = self.getDataFromFile(self.trainFeaturesFile)
        self.trainLabels = self.getDataFromFile(self.trainLabelsFile)
        self.valFeatures = self.getDataFromFile(self.valFeatuesFile)
        self.valLabels = self.getDataFromFile(self.valLabelsFile)

    def getDataFromFile(self, fileName):
        f = open(fileName)
        result = []
        
        for line in f:
            elem = [float(num) for num in line.strip().split(",")]
            result.append(elem)
        return result
    
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

    def testRF(self, tree_count = 50):
        #Training 
        forest = RandomForest(tree_count, self.trainFeatures, self.trainLabels, True)
        predictions = forest.predict(self.trainFeatures)
        acc = self.getAccuracy(predictions, self.trainLabels)
        print("Error rate on train set: " + str(acc * 100) + "%")
    
        #Validation     
        predictions = forest.predict(self.valFeatures)
        acc = self.getAccuracy(predictions, self.valLabels)
        print("Error rate on validation set: " + str(acc * 100) + "%")


if __name__ == "__main__":
    unittest.main()