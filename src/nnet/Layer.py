'''
Created on Dec 9, 2014

@author: martin
'''
from abc import ABCMeta, abstractmethod

class Layer(object):
    '''
    classdocs
    '''
    __metaclass__ = ABCMeta
    input = None
    output = None
    previous = None
    num_outputs = None
    outputshape = None
    params = None
    
    def __init__(self, params):
        '''
        Constructor
        '''
     
    @abstractmethod   
    def build(self):
        pass
    
#   @abstractmethod   
#   def serialize_params(self):
#       pass
    
#   @abstractmethod   
#   def deserialize_params(self):
#       pass