'''
Created on Nov 28, 2014

@author: martin
'''
import unittest
import theano.tensor as T
import theano
import os
import gzip
import cPickle
import numpy
import time
from classifiers import LogisticRegression
import sys

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        datadir = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "..",
            "data"
        )
        if not os.path.isdir(datadir):
            os.mkdir(datadir)
        new_path = os.path.join(datadir, dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval



class Test(unittest.TestCase):


    def setUp(self):
        # generate symbolic variables for input (x and y represent a
        # minibatch)
        self.x = T.matrix('x')  # data, presented as rasterized images
        self.y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
    
        # construct the logistic classifiers class
        # Each MNIST image has size 28*28
        self.classifier = LogisticRegression(input=self.x, n_in=28 * 28, n_out=10)
    
        # the cost we minimize during training is the negative log likelihood of
        # the model in symbolic format
        self.cost = self.classifier.negative_log_likelihood(self.y)

    def tearDown(self):
        pass
    
    def testsgd_optimization_mnist(self, learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
        """Demonstrate stochastic gradient descent optimization of a log-linear
        model
    
        This is demonstrated on MNIST.
    
        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)
    
        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
    
        :type dataset: string
        :param dataset: the path of the MNIST dataset file from
                     http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
        """
    
        datasets = load_data(dataset)
    
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
    
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'
    
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
    
    
        # compiling a Theano function that computes the mistakes that are made by
        # the model on a minibatch
        test_model = theano.function(
            inputs=[index],
            outputs=self.classifier.errors(self.y),
            givens={
                self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
    
        validate_model = theano.function(
            inputs=[index],
            outputs=self.classifier.errors(self.y),
            givens={
                self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
    
        # compute the gradient of cost with respect to theta = (W,b)
        g_W = T.grad(cost=self.cost, wrt=self.classifier.W)
        g_b = T.grad(cost=self.cost, wrt=self.classifier.b)
    
        # start-snippet-3
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        updates = [(self.classifier.W, self.classifier.W - learning_rate * g_W),
                   (self.classifier.b, self.classifier.b - learning_rate * g_b)]
    
        # compiling a Theano function `train_model` that returns the cost, but in
        # the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
            inputs=[index],
            outputs=self.cost,
            updates=updates,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
        # end-snippet-3
    
        ###############
        # TRAIN MODEL #
        ###############
        print '... training the model'
        # early-stopping parameters
        patience = 5000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
        # found
        improvement_threshold = 0.995  # a relative improvement of this much is
        # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
        # go through this many
        # minibatche before checking the network
        # on the validation set; in this case we
        # check every epoch
    
        best_validation_loss = numpy.inf
        test_score = 0.
        start_time = time.clock()
    
        done_looping = False
        epoch = 0
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):
    
                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index
    
                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i)
                                         for i in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
    
                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )
    
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)
    
                        best_validation_loss = this_validation_loss
                        # test it on the test set
    
                        test_losses = [test_model(i)
                                       for i in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)
    
                        print(
                            (
                                '     epoch %i, minibatch %i/%i, test error of'
                                ' best model %f %%'
                            ) %
                            (
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                test_score * 100.
                            )
                        )
    
                if patience <= iter:
                    done_looping = True
                    break
    
        end_time = time.clock()
        print(
            (
                'Optimization complete with best validation score of %f %%,'
                'with test performance %f %%'
            )
            % (best_validation_loss * 100., test_score * 100.)
        )
        print 'The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.1fs' % ((end_time - start_time)))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()