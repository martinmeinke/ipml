'''
Created on Dec 11, 2014

@author: martin
'''

#import pickle
import os
from os.path import split
import logging
import sPickle

logger = logging.getLogger(__name__)

class DatasetManager(object):
    '''
    classdocs
    '''

    def __init__(self, location="../../serialized_datasets"):
        '''
        Constructor
        '''
        self.dataset_location = location
        
    def dataset_available(self, datasetName):
        name = self.extract_last_component(datasetName)
        if os.path.exists(self.dataset_location+"/"+name):
            return True
        
        return False
    
    def extract_last_component(self, path):
        return split(path)[1]
    
    def store(self, dataset, datasetName):
        name = self.extract_last_component(datasetName)
        logger.info("Storing dataset: {}".format(name))
        sPickle.s_dump(dataset, open(self.dataset_location+"/"+name, "wb" ))
    
    def load(self, datasetName):
        name = self.extract_last_component(datasetName)
        logger.info("Loading dataset: {}".format(name))
        return sPickle.s_load(open(self.dataset_location+"/"+name, "rb" ))