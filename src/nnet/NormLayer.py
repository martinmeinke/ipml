'''
Created on Dec 9, 2014

@author: martin
'''
import numpy
import theano
from theano.tensor.nnet import conv
from nnet.Layer import Layer
import theano.tensor as T
from numpy import sqrt, prod, ones, floor, repeat, pi, exp, zeros, sum
from theano.tensor.nnet import conv2d
from theano import shared, _asarray, config
floatX = config.floatX


class NormLayer(Layer):

    """ Normalization layer """

    def __init__(self, input_shape, input=None, **kwargs):
        """
        method: "lcn", "gcn", "mean"
        LCN: local contrast normalization
            kwargs: 
                kernel_size=9, threshold=1e-4, use_divisor=True
        """

        self.input = input;
        self.input_shape = input_shape
        self.outputshape = input_shape
        self.params = []


    def lecun_lcn(self, X, kernel_size=3, threshold=1e-3, use_divisor=True):
        """
        Yann LeCun's local contrast normalization
        Orginal code in Theano by: Guillaume Desjardins
        """

        filter_shape = (1, 1, kernel_size, kernel_size)
        filters = gaussian_filter(kernel_size).reshape(filter_shape)
        filters = theano.shared(_asarray(filters, dtype=floatX), borrow=True)

        convout = conv2d(X, filters=filters, filter_shape=filter_shape,
                         border_mode='full')

        # For each pixel, remove mean of kernel_sizexkernel_size neighborhood
        mid = int(floor(kernel_size / 2.))
        new_X = X - convout[:, :, mid:-mid, mid:-mid]

        if use_divisor:
            # Scale down norm of kernel_sizexkernel_size patch
            sum_sqr_XX = conv2d(T.sqr(T.abs_(X)), filters=filters,
                                filter_shape=filter_shape, border_mode='full')

            denom = T.sqrt(sum_sqr_XX[:, :, mid:-mid, mid:-mid])
            per_img_mean = denom.mean(axis=[2, 3])
            divisor = T.largest(per_img_mean.dimshuffle(0, 1, 'x', 'x'), denom)
            divisor = T.maximum(divisor, threshold)

            new_X /= divisor

        return new_X  # T.cast(new_X, floatX)

    def build(self):
        # make 4D tensor out of 5D tensor -> (n_images, 1, height, width)
        # input_shape_4D = (input_shape[0]*input_shape[1]*input_shape[2], 1,
        #                     input_shape[3], input_shape[4])
        # input_4D = input.reshape(input_shape_4D, ndim=4)
        out = self.lecun_lcn(self.input)

        self.output = out.reshape(self.input_shape)

    def restore_params(self):
        pass


def gaussian_filter(kernel_shape):

    x = zeros((kernel_shape, kernel_shape), dtype='float32')

    def gauss(x, y, sigma=2.0):
        Z = 2 * pi * sigma ** 2
        return 1. / Z * exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = floor(kernel_shape / 2.)
    for i in xrange(0, kernel_shape):
        for j in xrange(0, kernel_shape):
            x[i, j] = gauss(i - mid, j - mid)

    return x / sum(x)
