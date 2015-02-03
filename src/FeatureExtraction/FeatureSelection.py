from rf.rnd_forest import RandomForest, ForestParams
from math import sqrt
from numpy import shape

class FeatureSelector(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def rf_get_k_best(self, k):
        print "test"
        data, features = shape(self.X)
        attr = int(sqrt(features))
        mtries = int(attr/2)
        subset = int(data*2/3)
        parms = ForestParams(attr, mtries, subset, -1, 30, 100, 10)
        rf = RandomForest(self.X, self.y, parms)
        return rf.get_k_best_variables(k)

        