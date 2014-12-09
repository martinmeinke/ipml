'''
Created on Dec 9, 2014

@author: martin
'''
import numpy

class Network(object):
    '''
    classdocs
    '''
    def __init__(self, params):
        '''
        Constructor
        '''
        self.hyperparameters = []
        self.layers = []
        
        self.rng = numpy.random.RandomState(23455)
    
    def add_layer(self, layer):
        #connect the new layer
        self.layers[-1].output = layer.input
        self.layers.append(layer)        
        
    def save_parameters(self):
        #pickle
        pass
        
    def load_parameters(self):
        pass
    