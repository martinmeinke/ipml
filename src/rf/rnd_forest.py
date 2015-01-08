import operator
import random
from rnd_tree import RandomTree 

class RandomForest(object):
    forest = None
    #TODO:optimize, not a fix number for subset, orig 100    
    #SAMPLE_SUBSET_SIZE = 1500
    SAMPLE_SUBSET_SIZE = 100
    
    def __init__(self, num_trees, training_features, training_labels, verbose = False):
        self.num_trees = num_trees
        self.forest = []
        self.verbose = verbose
        train_data = self.create_mapping(training_features, training_labels)
        self.generate_forest(train_data)
    
    
    def predict(self, features):
        predictions = []
        for feature in features:
            predictions.append(self.predict_one(feature))
            
        return predictions
    
    """
      classifies FEATURES using a majority vote from FOREST
    """
    def predict_one(self, feature):
        counts = {}
        for tree in self.forest:
            decision = tree.decide(feature)
            if counts.has_key(decision):
                counts[decision] += 1
            else:
                counts[decision] = 1
        sortedCounts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
        return sortedCounts[0][0]


    def generate_forest(self, data):
        for i in xrange(self.num_trees):
            data_subset = self.dict_subsample(data, self.SAMPLE_SUBSET_SIZE)
            attributes = [j for j in xrange(len(data.items()[0][0]))]
            self.forest.append(RandomTree(data_subset, attributes))
            if self.verbose:
                if i % 10 == 0:
                    print("Forest size: " + str(i))
    
    
    """
    Returns a random subset of each data set in ARGS of size SUB_SIZE with replacement
    """    
    def dict_subsample(self, data, sub_size):
        data_size = len(data)
        subset = {}
    
        for _ in xrange(sub_size):
            sample = data.items()[random.randrange(0, data_size)]
            subset[sample[0]] = sample[1][0]
    
        return subset
    
    def create_mapping(self, features, labels):
        """
        Returns a dictionary of keys(features) : values(labels)
        """
        result = {}
        if len(features) != len (labels):
            raise Exception("Keys and labels length do not match")
        for i in xrange(len(features)):
            result[tuple(features[i])] = labels[i]
        return result

        