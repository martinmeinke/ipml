'''
Created on Dec 9, 2014

@author: martin
'''
import numpy
import logging

logger = logging.getLogger(__name__)


class Network(object):

    cost = None
    errors = None
    gradients = None
    regularization_term = None

    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.model_params = []
        self.regularized_params = []
        self.regularized_params_weights = []
        self.layers = []

        self.rng = numpy.random.RandomState(None)

    def set_input(self, input_data):
        if len(self.layers) > 0:
            self.layers[0].input = input_data

    def add_layer(self, layer):
        # connect the new layer
        if len(self.layers) > 0:
            previous = self.layers[-1]
            layer.previous = previous

        self.layers.append(layer)

    def fix_structure(self):
        firstlayer = self.layers[0]
        firstlayer.build()

        logger.info("Created Layer {}\n\tOutput shape: {}".format(
            type(firstlayer), firstlayer.outputshape))

        # the first layer doesn't get it's input from a previous one
        for layer in self.layers[1:]:
            layer.input = layer.previous.output
            layer.build()
            logger.info("Created Layer {}\n\tOutput shape: {} Num outputs: {}".format(
                type(layer), layer.outputshape, layer.num_outputs))

        # create a list of all model parameters to be fit by gradient descent
        for layer in self.layers:
            self.model_params += layer.params
            self.regularized_params += layer.regularized_params
            self.regularized_params_weights += layer.regularized_params_weights

        logger.info("Regularized params: {}".format(self.regularized_params))

    def restore_params(self, serialized_plist):
        
        self.model_params = []
        for x in serialized_plist:
            self.model_params.append(x)
            
        param = 0

        for layer in self.layers:
            for lp in range(0, len(layer.params)):
                layer.params[lp] = self.model_params[param]
                param += 1

            layer.restore_params()

    def save_parameters(self):
        # pickle
        pass

    def load_parameters(self):
        pass