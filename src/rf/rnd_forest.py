import operator
import random
from rnd_tree import RandomTree, parallel_build_tree

from multiprocessing import Pool
import multiprocessing

class RandomForest(object):
    forest = None
    f_parms = None 
    
    def __init__(self, f_params, training_features, training_labels, verbose = False):
        self.f_parms = f_params
        self.forest = []
        self.verbose = verbose
        self.train_data = self.create_mapping(training_features, training_labels)

    
    def prepare_forest(self):
        for __ in xrange(self.f_parms.FOREST_SIZE):
            self.forest.append(RandomTree(self.f_parms))

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
            data_subset = self.dict_subsample(self.train_data, self.f_parms.SAMPLE_SUBSET_SIZE)
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
        for __ in xrange(self.f_parms.FOREST_SIZE):
            data_subset = self.dict_subsample(self.train_data, self.f_parms.SAMPLE_SUBSET_SIZE)
            attributes = [j for j in xrange(len(self.train_data.items()[0][0]))]
            args.append((self.f_parms, data_subset, attributes))
                            
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
            subset[sample[0]] = sample[1]
    
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

        