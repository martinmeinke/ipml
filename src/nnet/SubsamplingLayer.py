'''
Created on Dec 9, 2014

@author: martin
'''
from nnet.Layer import Layer
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
import numpy
import logging
from nnet.ConvolutionalLayer import ConvolutionalLayer

logger = logging.getLogger(__name__)


class SubsamplingLayer(Layer):
    '''
    classdocs
    '''

    def __init__(self, rng, poolsize=(2, 2), activation=0):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        self.rng = rng
        self.poolsize = poolsize
        self.activation = activation

    def compute_output_shape(self):
        if isinstance(self.previous, ConvolutionalLayer):
            batch_size = self.previous.outputshape[0]
            nkernels = self.previous.fshp[0]
            height = self.previous.outputshape[2] / self.poolsize[0]
            width = self.previous.outputshape[2] / self.poolsize[1]

            self.outputshape = (batch_size, nkernels, height, width)
        else:
            raise Exception("Unsupported Network Layout, Subsampling layer must follow a convolutional layer")

    def build(self):
        # the bias is a 1D tensor -- one bias per output feature map
        # TODO: init biases with one, when using relu?
        b_values = numpy.zeros((self.previous.num_output_featuremaps,), dtype=theano.config.floatX) # @UndefinedVariable
        self.b = theano.shared(value=b_values, borrow=False)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=self.input,
            ds=self.poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        # we add the bias term and squash it through our nonlinearity
        # self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        # TODO:use ReLU for debuging see: https://groups.google.com/forum/#!topic/theano-users/pbbddYetkgM
        # self.output = T.maximum(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'), 0)

        if self.activation == 0:
            logger.info("Activation function: TANH")
            self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        else:
            logger.info("Activation function: RELU")
            self.output = T.maximum(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'), 0)

        # no overlap (don't know if this is important)
        assert self.previous.num_outputs % numpy.prod(self.poolsize) == 0
        self.num_outputs = self.previous.num_outputs / numpy.prod(self.poolsize)

        if self.previous is None:
            self.num_output_featuremaps = 1
        else:
            self.num_output_featuremaps = self.previous.num_output_featuremaps

        self.compute_output_shape()
        # store parameters of this layer
        self.params = [self.b]

    def restore_params(self):
        self.b.container.data = self.params[0].container.data