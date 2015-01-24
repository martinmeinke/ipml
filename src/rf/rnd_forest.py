import operator
import random
from numpy import shape
from math import sqrt
from rnd_tree import RandomTree, parallel_build_tree
from multiprocessing import Pool
import multiprocessing
import logging

logger = logging.getLogger(__name__)

class ForestParams(object):
    #TODO:optimize, not a fix number for subset, orig 100    
    #SAMPLE_SUBSET_SIZE = 1500
    SAMPLE_SUBSET_SIZE = 100
    
    #if MIN_GAIN = None no minimum gain and go until MAX_TREE_DEPTH is reached
    #orig MIN_GAIN = 10e-5
    MIN_GAIN = 10e-6
    
    #TODO:optimize, size of feature subset orig 8
    NUM_ATTRIBUTES = 8
    #NUM_ATTRIBUTES = 8
    
    #TODO:optimize, threshold steps orig 10 
    NUM_THRES_STEPS = 10
    #NUM_THRES_STEPS = 10
    
    #maximum tries for testing different attributes to find best split
    MAX_TRIES = 80
    
    FOREST_SIZE = 100
    
    # if none train as long as data is available or MIN_GAIN is not reached with split
    MAX_TREE_DEPTH = None
    
    def __init__(self, num_intput, num_features):
        #self.NUM_ATTRIBUTES = num_features
        self.NUM_ATTRIBUTES = int(sqrt(num_features))*2
        self.SAMPLE_SUBSET_SIZE = int(num_intput*2/3)
        self.log()
        
    def log(self):
        msg = "RF parameters: "
        logger.info(msg)

class RandomForest(object):
    forest = None
    f_parms = None 
    num_data = None
    num_features = None
    
    def __init__(self, training_features, training_labels):
        self.num_data, self.num_features = shape(training_features)
        self.f_parms = ForestParams(self.num_data, self.num_features)
        self.train_data = training_features
        self.train_labels = training_labels        
        self.forest = []
    
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
        self.prepare_forest()
        for tree in self.forest:
            data_subset = self.gen_subset(self.num_data, self.f_parms.SAMPLE_SUBSET_SIZE)
            attributes = self.gen_subset(self.num_features, self.f_parms.NUM_ATTRIBUTES)         
            tree.build_tree(self.train_data, self.train_labels, data_subset, attributes)
            i+=1
            logger.info("Forest size: " + str(i))
                            
    
    def parallel_generate_forest(self):
        cores = multiprocessing.cpu_count()
        pool = Pool(processes=cores)
        args = []
        for __ in xrange(self.f_parms.FOREST_SIZE):
            data_subset = self.gen_subset(self.num_data, self.f_parms.SAMPLE_SUBSET_SIZE)
            attributes = self.gen_subset(self.num_features, self.f_parms.NUM_ATTRIBUTES)  
            args.append((self.f_parms, self.train_data, self.train_labels, data_subset, attributes))
                            
        self.forest = pool.map(parallel_build_tree, args)
        pool.close()
        pool.join()
    
    def gen_subset(self, size, sub_size):
        subset = []
        for _ in xrange(sub_size):
            sample_idx = random.randrange(0, size-1)
            subset.append(sample_idx)
        return subset


        