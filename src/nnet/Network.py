'''
Created on Dec 9, 2014

@author: martin
'''
import numpy
import logging
from imageio import create_samples, rgb2gray
import numpy 
import theano
import sys
import os
import theano.tensor as T
import time
from random import randrange
from imageio.helpers import plot_qimage_grayscale
import matplotlib.pyplot as plt
from imageio import DatasetManager

logger = logging.getLogger(__name__)

class Network(object):
    
    cost = None
    errors = None
    gradients = None
    
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        self.model_params = []
        self.layers = []
        
        self.rng = numpy.random.RandomState(23455)
        
    def set_input(self, input_data):
        if len(self.layers) > 0:
            self.layers[0].input = input_data
    
    def add_layer(self, layer):
        #connect the new layer
        if len(self.layers) > 0:
            previous = self.layers[-1]
            layer.previous = previous
       
        self.layers.append(layer)
        
    def fix_structure(self):
        self.layers[0].build()
        
        #the first layer doesn't get it's input from a previous one
        for layer in self.layers[1:]:
            layer.input = layer.previous.output
            layer.build()
            logger.info("Created Layer {}\n\tOutput shape: {}".format(type(layer), layer.outputshape))
            
        # create a list of all model parameters to be fit by gradient descent
        for layer in self.layers:
            self.model_params = self.model_params + layer.params
        
        
    def save_parameters(self):
        #pickle
        pass
        
    def load_parameters(self):
        pass


class Trainer(object):
    
    dataset_manager = DatasetManager("../../serialized_datasets")
    datasets = None
    
    # allocate symbolic variables for the data
    x = None
    y = None
    index = None
    
    n_train_batches = 0
    n_valid_batches = 0
    n_test_batches = 0
    
    n_train_samples = 0
    n_valid_samples = 0
    n_test_samples = 0
    
    batch_size = 250
    learning_rate = 0.03
    n_epochs = 200
    
    test_model = None
    validation_model = None
    training_model = None
    
    network = None
    
    #training parameters
    patience = 10000                # look as this many examples regardless
    patience_increase = 2           # wait this much longer when a new best is found
    improvement_threshold = 0.995   # a relative improvement of this much is considered significant
    validation_frequency = 1        # validate set every x minibatches
    
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    
    start_time = None
    
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
    
    def run_training(self):
        start_time = time.clock()
        
        plt.axis([0, self.n_epochs, 0, 0.6])
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Validation & Test Error')
        plt.ion()
        plt.show()
    
        #einmal alle bilder sehen = epoch
        epoch = 0
        done_looping = False
    
        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(self.n_train_batches):
    
                #jede batch eine Iteration?
                iter = (epoch - 1) * self.n_train_batches + minibatch_index
    
                if iter % 100 == 0:
                    print 'training @ iter = ', iter
                cost_ij = self.training_model(minibatch_index)
    
                if (iter + 1) % self.validation_frequency == 0:
    
                    # compute zero-one loss on validation set
                    validation_losses = [self.validation_model(i) for i
                                         in xrange(self.n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, self.n_train_batches,
                           this_validation_loss * 100.))
                    
                    plt.plot(epoch-1, this_validation_loss, 'bs')
                    plt.draw()
    
                    # if we got the best validation score until now
                    if this_validation_loss < self.best_validation_loss:
    
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < self.best_validation_loss *  \
                           self.improvement_threshold:
                            patience = max(self.patience, iter * self.patience_increase)
    
                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter
    
                        # test it on the test set
                        test_losses = [
                            self.test_model(i)
                            for i in xrange(self.n_test_batches)
                        ]
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, self.n_train_batches,
                               test_score * 100.))
                                        
                        plt.plot(epoch-1, test_score, 'g^')
                        plt.draw()
    
                if self.patience <= iter:
                    done_looping = True
                    break
    
        end_time = time.clock()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

    
    def get_input_size(self):
        return (self.batch_size, 1, 128, 128)
        
    def create_test_model(self):
        test_set_x, test_set_y = self.datasets[2]
        
        # create a function to compute the mistakes that are made by the model
        self.test_model = theano.function(
            [self.index],
            self.network.errors,
            givens={
                self.x: test_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: test_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )
        
    def create_validation_model(self):
        valid_set_x, valid_set_y = self.datasets[1]
        
        self.validation_model = theano.function(
            [self.index],
            self.network.errors,
            givens={
                self.x: valid_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: valid_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )
        
    def create_training_model(self):
        train_set_x, train_set_y = self.datasets[0]

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - self.learning_rate * grad_i)
            for param_i, grad_i in zip(self.network.model_params, self.network.gradients)
        ]
    
        self.training_model = theano.function(
            [self.index],
            self.network.cost,
            updates=updates,
            givens={
                self.x: train_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: train_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )
        
    def prepare_models(self):
        #use the negative log likelihood of the softmax layer
        self.network.cost = self.network.layers[-1].negative_log_likelihood(self.y)
        self.network.errors = self.network.layers[-1].errors(self.y)
        
        # create a list of gradients for all model parameters
        self.network.gradients = T.grad(self.network.cost, self.network.model_params)
        
        self.create_training_model()
        self.create_validation_model()
        self.create_test_model()
    
    def compute_training_parameters(self):
        train_set_x = self.datasets[0][0]
        valid_set_x = self.datasets[1][0]
        test_set_x  = self.datasets[2][0]
 
        # start-snippet-1
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y') # the labels are presented as 1D vector of
                                # [int] labels
                             
        self.n_train_samples = train_set_x.get_value(borrow=True).shape[0]
        self.n_valid_samples = valid_set_x.get_value(borrow=True).shape[0]
        self.n_test_samples = test_set_x.get_value(borrow=True).shape[0]
     
        logger.info("Set sizes:\n\tTrain: {}\n\tValid: {}\n\tTest: {}"
                    .format(self.n_train_samples, self.n_valid_samples, self.n_test_samples))
         
        self.n_train_batches = self.n_train_samples / self.batch_size
        self.n_valid_batches = self.n_valid_samples / self.batch_size
        self.n_test_batches = self.n_test_samples / self.batch_size
         
         
        #make sure, that the batch size isn't too big
        assert self.n_train_batches > 0 and self.n_valid_batches > 0 and self.n_test_batches > 0
     
        self.index = T.lscalar()  # index to a [mini]batch
      
    def prepare_data(self, dataset, partitioning):
      
        if self.dataset_manager.dataset_available(dataset):
            print('loading datasets...')
            sets = self.dataset_manager.load(dataset)
        else:
            print('creating datasets...')
            sets = create_samples(dataset, partitioning)
             
            #train_set[1].shape  (150,)
            #train_set[0].shape  (150,128,128,3)
            print('done')
             
            for i in range(len(sets)):
                #grayscale images
                sets[i] = (numpy.asarray(map(rgb2gray, sets[i][0])), sets[i][1])
                
                #sets[i] = (map(rgb2gray, sets[i][0]), sets[i][1])
                #sets[i] = (numpy.asarray(sets[i][0]), sets[i][1])
                 
                #=> need to create a flat representation of the image
                reshape_dims = (sets[i][0].shape[0],128*128)
                sets[i] = (numpy.reshape(sets[i][0].astype(numpy.float32) / 255, reshape_dims), sets[i][1]);
            
            self.dataset_manager.store(sets, dataset)
            
        train_set, valid_set, test_set = sets
     
        if logger.getEffectiveLevel() is logging.DEBUG:
            #view a couple of images to confirm that the class lables are correct
            for s in sets:
                for i in range(5):
                    random_index = randrange(0,len(s[0]))
                    plot_qimage_grayscale(s[0][random_index], s[1][random_index])
                 
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
             
            borrow has something todo with copying the numpy array or not
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
     
        #move the data to shared theano datasets
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)
     
        datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        
        self.datasets = datasets
        
        self.compute_training_parameters()
