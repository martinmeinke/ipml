import os
import sys
import logging
import numpy as np
from FeatureProvider import FeatureProvider
from FeatureExtraction.FeatureClass import FeatureExtractor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

BASEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../") 

def main():
    logging.basicConfig(stream=sys.stdout, format='[%(levelname)s|%(asctime)s] %(message)s', level=logging.DEBUG)
    logging.info("Loading data")
    featurepath = os.path.join(BASEDIR, "saved/extracted_features.8000.500.gz")
    datamax = 5000 
    
    fp = FeatureProvider(None, FeatureExtractor())
    fp.loadFromFile(featurepath, datamax)
        
    scaler = StandardScaler()
    X = scaler.fit_transform(fp.TrainData)
    Y = fp.TrainLabels.A1
    
    logging.info("Initialize grid search")
    C_range = 10.0 ** np.arange(-2, 9)
    gamma_range = 10.0 ** np.arange(-5, 4)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedKFold(y=Y, n_folds=3)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    logging.info("Fitting grid...")
    grid.fit(X, Y)
    
    print "The best classifier is: ", grid.best_estimator_
    logging.info("The best classifier is: %s", str(grid.best_estimator_))

if __name__ == '__main__':
    main()
    