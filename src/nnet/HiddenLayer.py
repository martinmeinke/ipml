'''
Created on Nov 28, 2014

@author: martin
'''

import numpy
import theano
import theano.tensor as T
from nnet.Layer import Layer
from nnet.SubsamplingLayer import SubsamplingLayer

# start-snippet-1
class HiddenLayer(Layer):
    def __init__(self, rng, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.rng = rng
        self.n_out = n_out
        self.W = W
        self.b = b
        self.activation = activation
        
    def build(self):    
        n_in = self.previous.num_outputs

        if self.W is None:
            W_values = numpy.asarray(
                self.rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + self.n_out)),
                    high=numpy.sqrt(6. / (n_in + self.n_out)),
                    size=(n_in, self.n_out)
                ),
                dtype=theano.config.floatX  # @UndefinedVariable
            )
            
            if self.activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            self.W = theano.shared(value=W_values, name='W', borrow=True)

        if self.b is None:
            b_values = numpy.zeros((self.n_out,), dtype=theano.config.floatX)  # @UndefinedVariable
            self.b = theano.shared(value=b_values, name='b', borrow=True)

        #convert input into a flat representation 
        #(no more depth in feature maps, or 2d images - just a
        # 1 d layer of regular neurons
        if isinstance(self.previous, SubsamplingLayer):
            self.input = self.input.flatten(2)
            
        lin_output = T.dot(self.input, self.W) + self.b
        
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        
        self.num_outputs = self.n_out
        
        if self.previous == None:
            self.num_output_featuremaps = 1
        else:
            self.num_output_featuremaps = self.previous.num_output_featuremaps

        # parameters of the model
        self.params = [self.W, self.b]
