'''
Created on Nov 28, 2014

@author: martin
'''
import unittest
import regression
import theano.tensor as T


class Test(unittest.TestCase):


    def setUp(self):
        # generate symbolic variables for input (x and y represent a
        # minibatch)
        x = T.matrix('x')  # data, presented as rasterized images
        y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
    
        # construct the logistic regression class
        # Each MNIST image has size 28*28
        classifier = regression.LogisticRegression(input=x, n_in=28 * 28, n_out=10)

        # the cost we minimize during training is the negative log likelihood of
        # the model in symbolic format
        cost = classifier.negative_log_likelihood(y)

    def tearDown(self):
        pass


    def testName(self):
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()