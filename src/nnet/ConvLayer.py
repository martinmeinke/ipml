'''
Created on Dec 9, 2014

@author: martin
'''
import numpy
import theano
from theano.tensor.nnet import conv
from nnet.Layer import Layer


class ConvLayer(Layer):

    '''
    classdocs
    '''

    def __init__(self, rng, fshp, imgshp=None, batch_size=0, inpt=None, regularizer_weight=1):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        """

        self.rng = rng
        self.fshp = fshp
        self.imgshp = imgshp
        self.batch_size = batch_size
        self.inpt = inpt
        self.reg_weight = regularizer_weight

    def compute_output_shape(self):
        # TODO
        # verify that the window fits..
        nkerns = self.fshp[0]
        width = self.imgshp[2] - self.fshp[2] + 1
        height = self.imgshp[3] - self.fshp[3] + 1
        self.outputshape = (self.batch_size, nkerns, height, width)

    def build(self):
        if self.imgshp is None:
            # we need: (batch size, num input feature maps, image height, image
            # width)
            self.imgshp = self.previous.outputshape

        assert self.imgshp[1] == self.fshp[1]

        if self.inpt is not None:
            self.input = self.inpt

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(self.fshp[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        # the 4 was numpy.prod(poolsize) before
        fan_out = (self.fshp[0] * numpy.prod(self.fshp[2:]) / 4)
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))

        self.W = theano.shared(
            numpy.asarray(
                self.rng.uniform(low=-W_bound, high=W_bound, size=self.fshp),
                dtype=theano.config.floatX  # @UndefinedVariable
            ),
            borrow=False
        )

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=self.input,
            filters=self.W,
            filter_shape=self.fshp,
            image_shape=self.imgshp
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        # we add the bias term and squash it through our nonlinearity
        self.output = conv_out
        conv_out_pixels = (
            self.imgshp[2] - self.fshp[2] + 1) \
            * (self.imgshp[3] - self.fshp[3] + 1)

        # TODO: compute via outputshape
        self.num_outputs = conv_out_pixels * self.fshp[0]

        self.num_output_featuremaps = self.fshp[0]

        self.compute_output_shape()

        # store parameters of this layer
        self.params = [self.W]
        self.regularized_params = [self.W]
        self.regularized_params_weights = [self.reg_weight]

    def restore_params(self):
        self.W.set_value(self.params[0].container.data)
        # self.W.container.data = self.params[0].container.data
