'''
Created on Dec 9, 2014

@author: martin
'''
from nnet.Layer import Layer
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from imageio import create_samples, rgb2gray

class SubsamplingLayer(Layer):
    '''
    classdocs
    '''

    def __init__(self, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX) # @UndefinedVariable
        self.b = theano.shared(value=b_values, borrow=True)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=self.input,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        # we add the bias term and squash it through our nonlinearity
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        #TODO:use ReLU for debuging see: https://groups.google.com/forum/#!topic/theano-users/pbbddYetkgM
        #self.output = T.maximum(self.output, 0)
        
        # store parameters of this layer
        self.params = [self.b]
        