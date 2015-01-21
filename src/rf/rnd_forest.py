import operator
import random
from rnd_tree import RandomTree, parallel_build_tree

from multiprocessing import Pool
import multiprocessing

class RandomForest(object):
    forest = None
    #TODO:optimize, not a fix number for subset, orig 100    
    #SAMPLE_SUBSET_SIZE = 1500
    SAMPLE_SUBSET_SIZE = 100
    
    MIN_GAIN = 10e-5
    #TODO:optimize, size of feature subset orig 8
    #NUM_ATTRIBUTES = 20
    NUM_ATTRIBUTES = 8
    #TODO:optimize, threshold steps orig 10 
    #NUM_THRES_STEPS = 40
    NUM_THRES_STEPS = 10
    
    #maximum tries for testing different attribute sets to find best split
    MAX_TRIES = 10
    
    def __init__(self, num_trees, training_features, training_labels, verbose = False):
        self.num_trees = num_trees
        self.forest = []
        self.verbose = verbose
        self.train_data = self.create_mapping(training_features, training_labels)

    
    def prepare_forest(self):
        for __ in xrange(self.num_trees):
            self.forest.append(RandomTree(self.MIN_GAIN, self.NUM_ATTRIBUTES, self.NUM_THRES_STEPS, self.MAX_TRIES))

    def predict(self, features):
        predictions = []
        for feature in features:
            predictions.append(self.predict_one(feature))
        return predictions
    
    """
      classifies FEATURE using a majority vote from FOREST
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


    def generate_forest(self):
        i=0
        args = []
        self.prepare_forest()
        for tree in self.forest:
            data_subset = self.dict_subsample(self.train_data, self.SAMPLE_SUBSET_SIZE)
            attributes = [j for j in xrange(len(self.train_data.items()[0][0]))]
            tree.build_tree(data_subset, attributes)
            i+=1
            if self.verbose:
                if i % 10 == 0:
                    print("Forest size: " + str(i))
                            
    
    def parallel_generate_forest(self):
        cores = multiprocessing.cpu_count()
        pool = Pool(processes=cores)
        args = []
        for __ in xrange(self.num_trees):
            data_subset = self.dict_subsample(self.train_data, self.SAMPLE_SUBSET_SIZE)
            attributes = [j for j in xrange(len(self.train_data.items()[0][0]))]
            args.append(((self.MIN_GAIN, self.NUM_ATTRIBUTES, self.NUM_THRES_STEPS, self.MAX_TRIES), data_subset, attributes))
                            
        self.forest = pool.map(parallel_build_tree, args)
        pool.close()
        pool.join()
    
    
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

        