'''
Created on Nov 28, 2014

@author: martin
'''

import numpy
import theano
import theano.tensor as T
from nnet.Layer import Layer
from nnet.SubsamplingLayer import SubsamplingLayer
from nnet.ConvLayer import ConvLayer
import logging

logger = logging.getLogger(__name__)


# start-snippet-1
class HiddenLayer(Layer):
    def __init__(self, rng, n_out, W=None, b=None,
                 activation=0, regularizer_weight=1):
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
        self.reg_weight = regularizer_weight
    
    def build(self):
        n_in = self.previous.num_outputs

        W_bound = numpy.sqrt(6. / (n_in + self.n_out))

        if self.W is None:
            W_values = numpy.asarray(
                self.rng.uniform(
                    low=-W_bound,
                    high=W_bound,
                    size=(n_in, self.n_out)
                ),
                dtype=theano.config.floatX  # @UndefinedVariable
            )

            # some guy found out this works better for sigmoids
            # if self.activation == theano.tensor.nnet.sigmoid:
            if self.activation == 2:
                W_values *= 4

            self.W = theano.shared(value=W_values, name='W', borrow=False)

        if self.b is None:
            #relus are useless in ther 0-zone
            if self.activation == 1:
                b_values = numpy.ones((self.n_out,), dtype=theano.config.floatX)  # @UndefinedVariable
            else:
                b_values = numpy.zeros((self.n_out,), dtype=theano.config.floatX)  # @UndefinedVariable
            self.b = theano.shared(value=b_values, name='b', borrow=False)

        # convert input into a flat representation
        # (no more depth in feature maps, or 2d images - just a
        # 1 d layer of regular neurons
        if isinstance(self.previous, SubsamplingLayer) or isinstance(self.previous, ConvLayer):
            self.input = self.input.flatten(2)

        lin_output = T.dot(self.input, self.W) + self.b

#         self.output = (
#             lin_output if self.activation is None
#             else self.activation(lin_output)
#         )

        if self.activation == 0:
            logger.info("Activation function: TANH")
            self.output = T.tanh(lin_output)
        else:
            logger.info("Activation function: RELU")
            self.output = T.maximum(lin_output, 0)

        self.num_outputs = self.n_out

        if self.previous is None:
            self.num_output_featuremaps = 1
        else:
            self.num_output_featuremaps = self.previous.num_output_featuremaps

        # parameters of the model
        self.params = [self.W, self.b]
        self.regularized_params = [self.W]
        self.regularized_params_weights = [self.reg_weight]

    def restore_params(self):
        # self.W.container.data = self.params[0].container.data
        # self.b.container.data = self.params[1].container.data
        self.W.set_value(self.params[0].container.data)
        self.b.set_value(self.params[1].container.data)
