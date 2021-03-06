import operator
import random
import numpy as np
from numpy import shape
from math import sqrt
from rnd_tree import RandomTree, parallel_build_tree
from multiprocessing import Pool
import multiprocessing
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_classif

logger = logging.getLogger(__name__)

class ForestParams(object):
    #TODO:optimize, not a fix number for subset, orig 100    
    #SAMPLE_SUBSET_SIZE = 1500
    SAMPLE_SUBSET_SIZE = 100
    
    #if MIN_GAIN = None no minimum gain and go until MAX_TREE_DEPTH is reached
    #orig MIN_GAIN = 10e-5
    MIN_GAIN = 1.5*10e-3
    
    #TODO:optimize, size of feature subset orig 8
    NUM_ATTRIBUTES = 8
    #NUM_ATTRIBUTES = 8
    
    #TODO:optimize, threshold steps orig 10 
    NUM_THRES_STEPS = 40
    #NUM_THRES_STEPS = 10
    
    #maximum tries for testing different attributes to find best split
    MAX_TRIES = 15
    
    FOREST_SIZE = 500
    
    # if negative train as long as data is available or MIN_GAIN is not reached with split
    MAX_TREE_DEPTH = None
    
    def __init__(self, num_attr, max_tries, subset, min_gain, thres_steps, forest_size, depth):
        #self.NUM_ATTRIBUTES = num_features
        #self.NUM_ATTRIBUTES = num_features#*70/100
        #self.MAX_TRIES = num_features/5
        #self.SAMPLE_SUBSET_SIZE = int(num_intput*1/3)
        self.NUM_ATTRIBUTES = num_attr
        self.MAX_TRIES = max_tries
        self.SAMPLE_SUBSET_SIZE = subset
        self.MIN_GAIN = min_gain
        self.NUM_THRES_STEPS = thres_steps
        self.FOREST_SIZE = forest_size
        self.MAX_TREE_DEPTH = depth
        self.log()
        
    def log(self):
        msg = "RF parameters: n_attr=%d, max_tries=%d, subset=%d, min_gain=%f, thres_steps=%d, forest_size=%d, max_depth=%d"% (self.NUM_ATTRIBUTES, self.MAX_TRIES, self.SAMPLE_SUBSET_SIZE, self.MIN_GAIN, self.NUM_THRES_STEPS, self.FOREST_SIZE, self.MAX_TREE_DEPTH)
        print msg
        logger.info(msg)

class RandomForest(object):
    forest = None
    f_parms = None 
    num_data = None
    num_features = None
    used_features = []
    kbest_features = None
    
    
    def __init__(self, training_features, training_labels, params):
        self.num_data, self.num_features = shape(training_features)
        self.f_parms = params
        #self.select_features(training_features, training_labels)
        self.train_data = training_features.tolist()
        self.train_labels = training_labels.tolist()
        self.forest = []
    
    
    #TODO:remove because is sklearn implementation 
    def select_features(self, X, y):
        estimator = SelectKBest(chi2, 700)
        estimator.fit(X, y.A1)
        support_mask = estimator._get_support_mask()
        features = []
        i = 0
        for feature in support_mask:
            if(feature):
                features.append(i)
            i +=1
        self.kbest_features = features
        self.num_features = len(features)
        
    #generate random forest and calculate variable importance and return the K best ones
    def get_k_best_variables(self, k):
        msg = 'calculate important features with random forest classifier'
        print msg
        logger.info(msg)
        self.parallel_generate_forest(True)
        var_imp = self.forest[0].var_importance
        used_var = self.forest[0].used_features
        for i in xrange(1, self.f_parms.FOREST_SIZE):
            var_imp = np.add(var_imp, self.forest[i].var_importance)
            used_var = np.add(used_var, self.forest[i].used_features)
        var_imp = np.divide(var_imp, used_var)
        #sort variables respecting importance
        var_sorted = [i[0] for i in sorted(enumerate(var_imp.tolist()), key=lambda x:x[1], reverse=True)]
        if(k > len(var_sorted) and k < 0):
            msg = "Error: k =" + str(k) + " but only " +str(len(var_sorted)) + " are available"
            logger.log(1, msg)
            return
        return var_sorted[:k]
        
    def set_kbest(self, kbest):
        self.kbest_features = kbest
        self.num_features = len(kbest)
        
    def prepare_forest(self):
        for __ in xrange(self.f_parms.FOREST_SIZE):
            self.forest.append(RandomTree(self.f_parms))

    def predict(self, features):
        print "predict"
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
            try:
                counts[decision] += 1
            except:
                counts[decision] = 1
        sortedCounts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
        return sortedCounts[0][0]


    def generate_forest(self, oob=False):
        print "generate forest"
        i=0
        self.prepare_forest()
        for tree in self.forest:
            data_subset, oob = self.gen_subset_exclusive(self.num_data, self.f_parms.SAMPLE_SUBSET_SIZE)
            attributes = self.gen_feature_set(self.num_features, self.f_parms.NUM_ATTRIBUTES)[0]  
            if(oob):       
                tree.build_tree(self.train_data, self.train_labels, data_subset, attributes, oob)
            else:
                tree.build_tree(self.train_data, self.train_labels, data_subset, attributes)
            i+=1
            logger.info("Forest size: " + str(i))
                            
    
    def parallel_generate_forest(self, oob = False):
        print "generate forest"
        cores = multiprocessing.cpu_count()
        pool = Pool(processes=cores)
        args = []
        
        for __ in xrange(self.f_parms.FOREST_SIZE):
            data_subset, oob = self.gen_subset_exclusive(self.num_data, self.f_parms.SAMPLE_SUBSET_SIZE)
            attributes = self.gen_feature_set(self.num_features, self.f_parms.NUM_ATTRIBUTES)[0]
            if(oob):
                args.append((self.f_parms, self.train_data, self.train_labels, data_subset, attributes, oob))
            else:
                args.append((self.f_parms, self.train_data, self.train_labels, data_subset, attributes))
                            
        self.forest = pool.map(parallel_build_tree, args)
        pool.close()
        pool.join()
    
    # generate subset with putting back
    def gen_subset(self, size, sub_size):
        subset = []
        for _ in xrange(sub_size):
            sample_idx = random.randrange(0, size-1)
            subset.append(sample_idx)
        return subset
    
    # generate subset without putting back
    def gen_subset_exclusive(self, size, sub_size, feature_exclusive=False):
        subset = []
        orig_list=[]
        for i in xrange(size):
            if(feature_exclusive):
                if i not in self.used_features:
                    orig_list.append(i)
            else:
                orig_list.append(i)
        if(len(orig_list) < sub_size):
            orig_list = [j for j in xrange(size)]
        random.shuffle(orig_list)
        
        for _ in xrange(sub_size):
            sample_idx = orig_list.pop()
            if(feature_exclusive):
                self.used_features.append(sample_idx)
            subset.append(sample_idx)
        return (subset, orig_list)
    
    def gen_feature_set(self, size, sub_size):
        
        if(self.kbest_features is None):
            return self.gen_subset_exclusive(size, sub_size, True)
        else:
            subset = []
            orig_list = list(self.kbest_features)
            random.shuffle(orig_list)
            for _ in xrange(sub_size):
                sample_idx = orig_list.pop()
                subset.append(sample_idx)
            return (subset, orig_list)
            
        


        