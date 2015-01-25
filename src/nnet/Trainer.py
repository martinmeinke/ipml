import numpy
import logging
from imageio import create_samples, rgb2gray
import theano
import sys
import math
import os
import theano.tensor as T
import time
from random import randrange
from imageio.helpers import plot_qimage_grayscale
import matplotlib.pyplot as plt
from imageio import DatasetManager
from imageio import helpers
from numpy import int8

logger = logging.getLogger(__name__)


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


def shared_dataset_x(data_x, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.

        borrow has something todo with copying the numpy array or not
        """

        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x


class Trainer(object):

    dataset_manager = DatasetManager("../../serialized_datasets")
    datasets = None

    img_width = 0
    img_height = 0

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

    learning_rate_t0 = 1

    epoch = 0

    test_model = None
    validation_model = None
    train_validation_model = None
    training_model = None

    network = None

    # training parameters
    patience = 30000                # run for this many iterations regardless
    # wait this much longer when a new best is found
    patience_increase = 2
    # a relative improvement of this much is considered significant
    improvement_threshold = 0.995
    validation_frequency = 0        # validate set every x minibatches

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0

    start_time = None

    PLOT_SYMBOLS = {"Training": 'bs', "Validation": 'rs', "Test": 'g^'}

    '''
    classdocs
    '''

    def __init__(self, lr_lambda=0.002, batch_size=200, n_epochs=120, lr_decay=True):
        '''
        Constructor
        '''
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate_lambda = lr_lambda
        self.learning_rate = lr_lambda
        self.learning_rate_decay = lr_decay

    def update_learning_rate(self, t, decrease_speed=0.15):
        '''
        lowers the learning rate, as the learning proceeds
        '''
        self.learning_rate = max(helpers.lr_decay(self.learning_rate_lambda,
                                                  self.learning_rate_t0,
                                                  decrease_speed, t), 0.0001)

        logger.info("Current learning rate: {}".format(
                    self.learning_rate))

    def plot_datapoint(self, series, y, epoch):
        '''
        Plots a the given datapoint y to the main figure(1), 
        using the epoch as an x value
        '''
        plt.plot(self.epoch, y, self.PLOT_SYMBOLS[
                 series], label=series if self.epoch == 1 else "")

    def run_training(self):
        start_time = time.clock()
        # TODO: plot lr aswell
#         gs = gridspec.GridSpec(2, 1, width_ratios=[3, 1]) 
#         ax0 = plt.subplot(gs[0])
#         ax0.plot(x, y)
#         ax1 = plt.subplot(gs[1])

        plt.figure(1)
        plt.axis([0, self.n_epochs, 0, 0.6])
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Training, Validation & Test Error')

        plt.ion()
        plt.show()

        # einmal alle bilder sehen = epoch
        done_looping = False

        try:
            while (self.epoch < self.n_epochs) and (not done_looping):

                helpers.plot_kernels(self.network.layers,
                                     layerids=(0, 2, 4),
                                     figids=(2, 3, 4))

                if self.learning_rate_decay:
                    # decrease the learning rate to yield better results
                    self.update_learning_rate(self.epoch)

                # stop the training, if a key was pressed during last iteration
                if helpers.check_break():
                    break

                self.epoch = self.epoch + 1
                for minibatch_index in xrange(self.n_train_batches):

                    # 1 batch = 1 iteration?
                    iternum = (self.epoch - 1) * \
                        self.n_train_batches + minibatch_index

                    # output the iteration number from time to time
                    if iternum % 10 == 0:
                        logger.info("training @ iternum = {}".format(iternum))

                    cost_ij, regularization = self.training_model(minibatch_index)

                    # monitor changes in one specific minibatch
                    if minibatch_index == 0:
                        # only for debugging
                        logger.debug(
                            "Regularization cost: {}".format(regularization))
                        logger.info("Training model cost: {}".format(cost_ij))

                    if (iternum + 1) % self.validation_frequency == 0:

                        training_losses = [self.train_validation_model(i) for i
                                           in xrange(self.n_train_batches)]

                        logger.debug(
                            "Training losses: {}".format(training_losses))

                        # compute zero-one loss on validation set
                        validation_losses = [self.validation_model(i) for i
                                             in xrange(self.n_valid_batches)]

                        logger.debug(
                            "Training losses: {}".format(validation_losses))

                        this_training_loss = numpy.mean(training_losses)
                        this_validation_loss = numpy.mean(validation_losses)

                        print("epoch {}, minibatch {}/{}, training_error: {}, "
                              "validation error: {}".format(
                                  self.epoch, minibatch_index +
                                  1, self.n_train_batches,
                                  this_training_loss * 100.,
                                  this_validation_loss * 100.))

                        self.plot_datapoint(
                            "Training", this_training_loss, self.epoch)
                        self.plot_datapoint(
                            "Validation", this_validation_loss, self.epoch)
                        plt.draw()

                        # if we got the best validation score until now
                        if this_validation_loss < self.best_validation_loss:

                            # improve patience if loss improvement is good
                            # enough
                            if this_validation_loss < self.best_validation_loss *  \
                               self.improvement_threshold:
                                self.patience = max(
                                    self.patience, iternum * self.patience_increase)

                            # save best validation score and iteration number
                            self.best_validation_loss = this_validation_loss
                            self.best_iter = iternum

                            # test it on the test set
                            test_losses = [
                                self.test_model(i)[1]
                                for i in xrange(self.n_test_batches)
                            ]

                            # test it on the test set
                            misclassified_images = [
                                self.test_model(i)[0]
                                for i in xrange(self.n_test_batches)
                            ]

                            helpers.plot_misclassified_images(
                                misclassified_images, self.datasets[2], edgelen=self.img_width)

                            self.test_score = numpy.mean(test_losses)
                            print(('     epoch %i, minibatch %i/%i, test error of '
                                   'best model %f %%') %
                                  (self.epoch, minibatch_index + 1, self.n_train_batches,
                                   self.test_score * 100.))

                            logger.info(
                                "Patience: {}, iter: {}".format(self.patience, iternum))

                            # self.plot_datapoint("Test", self.test_score)
                            plt.draw()
                    # break training, afrer we've been patient enough
                    if self.patience <= iternum:
                        self.patience += 1
                        done_looping = True
                        break

        except KeyboardInterrupt:
            pass

        plt.legend(loc='upper right')
        plottxt = "Training, Validation & Test Error\nLearning Rate: {}, Batch Size: {}, Best Test: {}\n Train set size: {}". \
            format(self.learning_rate, self.batch_size, self.test_score, self.n_train_samples)
        plt.title(plottxt)

        plotname = "../../plots/{}.png".format(time.strftime("%c"))
        logger.info("Saving plot: {}".format(plotname))
        plt.savefig(plotname)

        end_time = time.clock()
        logger.info('Optimization complete.')
        logger.info('Best validation score of %f %% obtained at iteration %i, '
                    'with test performance %f %%' %
                    (self.best_validation_loss * 100., self.best_iter + 1, self.test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

    def evaluate_test_set(self):
        # test it on the test set
        labels = [
            self.prediction_model(i)[0]
            for i in xrange(self.n_test_batches)
        ]

        labels = [item for sublist in labels for item in sublist]
        return labels

    def get_input_size(self):
        return (self.batch_size, 1, self.img_width, self.img_height)

    def create_test_model(self):
        test_set_x, test_set_y = self.datasets[2]

        # create a function to compute the mistakes that are made by the model
        self.test_model = theano.function(
            [self.index],
            [self.network.errorslist, self.network.errors],
            givens={
                self.x: test_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: test_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )

    def create_prediction_model(self):
        test_set_x = self.datasets[0]

        # create a function to compute the mistakes that are made by the model
        self.prediction_model = theano.function(
            [self.index],
            [self.network.layers[-1].y_pred],
            givens={
                self.x: test_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size]
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

    def create_train_validation_model(self):
        train_set_x, train_set_y = self.datasets[0]

        self.train_validation_model = theano.function(
            [self.index],
            self.network.errors,
            givens={
                self.x: train_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: train_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )

    def create_training_model(self):
        train_set_x, train_set_y = self.datasets[0]

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        # TODO: different lr for different layer types
        updates = [
            (param_i, param_i - self.learning_rate * grad_i)
            for param_i, grad_i in zip(self.network.model_params, self.network.gradients)
        ]

        self.training_model = theano.function(
            [self.index],
            [self.network.cost, self.network.regularization_term],
            updates=updates,
            givens={
                self.x: train_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: train_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )

    def prepare_models(self, run_mode=0):
        # use the negative log likelihood of the softmax layer
        logreg_layer = self.network.layers[-1]
        self.network.regularization_term = logreg_layer.regularization(
            self.network.regularized_params)
        self.network.cost = logreg_layer.nll(
            self.y) + logreg_layer.regularization(self.network.regularized_params)
        self.network.errorslist = self.network.layers[-1].errorslist(self.y)
        self.network.errors = self.network.layers[-1].errors(self.y)

        # create a list of gradients for all model parameters
        self.network.gradients = T.grad(
            self.network.cost, self.network.model_params)

        if run_mode == 0:
            self.create_training_model()
            self.create_validation_model()
            self.create_train_validation_model()
            self.create_test_model()
        elif run_mode == 1:
            self.create_prediction_model()

    def compute_training_parameters(self):
        train_set_x = self.datasets[0][0]
        valid_set_x = self.datasets[1][0]
        test_set_x = self.datasets[2][0]

        # start-snippet-1
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
        # [int] labels

        self.n_train_samples = train_set_x.get_value(borrow=True).shape[0]
        self.n_valid_samples = valid_set_x.get_value(borrow=True).shape[0]
        self.n_test_samples = test_set_x.get_value(borrow=True).shape[0]

        logger.info("Set sizes:\n\tTrain: {}\n\tValid: {}\n\tTest: {}"
                    .format(self.n_train_samples, self.n_valid_samples, self.n_test_samples))

        self.n_train_batches = self.n_train_samples / self.batch_size
        self.n_valid_batches = self.n_valid_samples / self.batch_size
        self.n_test_batches = self.n_test_samples / self.batch_size

        # make sure, that the batch size isn't too big
        assert self.n_train_batches > 0 and self.n_valid_batches > 0 and self.n_test_batches > 0

        self.index = T.lscalar()  # index to a [mini]batch

    def prepare_kaggle_test_data(self, dataset):
        if self.dataset_manager.dataset_available(dataset):
            logger.info('loading kaggle test dataset...')
            test_set = self.dataset_manager.load(dataset, False)

            print test_set.shape

            el = int(math.sqrt(test_set.shape[1]))
            self.img_width = el
            self.img_height = el
        else:
            logger.info('creating kaggle test dataset...')
            filenames = os.listdir(dataset)
            filenames = sorted(filenames, key=lambda x: int(x.split(".")[0]))

            images = map(lambda x: helpers.convert_image_representation(dataset + '/' + x, False), filenames)
            test_set = numpy.asarray(map(lambda x: x[0], images))

            self.img_width = test_set.shape[1]
            self.img_height = test_set.shape[2]

            # grayscale images
            gray = map(rgb2gray, test_set)
            test_set = numpy.asarray(gray)

            flat_img_len = self.img_width * self.img_height

            # need to create a flat representation of the image
            reshape_dims = (test_set.shape[0], flat_img_len)
            test_set = numpy.reshape(test_set.astype(numpy.float16) / 255, reshape_dims)

            print test_set.shape
            self.dataset_manager.store(test_set, dataset, False)

        # move the data to shared theano datasets
        test_set_x = shared_dataset_x(test_set)

        self.datasets = [test_set_x]

        # start-snippet-1
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
        # [int] labels

        self.n_test_samples = test_set_x.get_value(borrow=True).shape[0]

        logger.info("Set sizes: Test: {}".format(self.n_test_samples))

        self.n_test_batches = self.n_test_samples / self.batch_size

        # make sure, that the batch size isn't too big
        assert self.n_test_batches > 0
        self.index = T.lscalar()  # index to a [mini]batch

    def prepare_training_data(self, dataset, partitioning=(6, 2, 2), partial=100, train_enhanced=False):
        if partial > 100 or partial < 1:
            raise ValueError("partial must lie in between 1 and 100")

        stored_ds_name = "{}_{}".format(dataset, partial)

        if self.dataset_manager.dataset_available(stored_ds_name):
            logger.info('loading datasets...')
            sets = self.dataset_manager.load(stored_ds_name)

            train_set, valid_set, test_set = sets
            el = int(math.sqrt(train_set[0][1].shape[0]))
            self.img_width = el
            self.img_height = el
        else:
            logger.info('creating datasets...')
            sets = create_samples(dataset, partitioning, partial, train_enhanced=train_enhanced)

            # train_set[1].shape  (150,)
            # train_set[0].shape  (150,128,128,3)
            self.img_width = sets[0][0].shape[1]
            self.img_height = sets[0][0].shape[2]

            # for each set
            for i in range(len(sets)):
                # grayscale images
                gray = map(rgb2gray, sets[i][0])
                sets[i] = (numpy.asarray(gray), numpy.asarray(sets[i][1], int8))

                flat_img_len = self.img_width * self.img_height

                # need to create a flat representation of the image
                reshape_dims = (sets[i][0].shape[0], flat_img_len)
                sets[i] = (
                    numpy.reshape(sets[i][0].astype(numpy.float16) / 255,
                                  reshape_dims),
                    sets[i][1])

            train_set, valid_set, test_set = sets
            self.dataset_manager.store(sets, stored_ds_name)

        logger.info(
            "Img dimensions: {}/{}".format(self.img_width, self.img_height))

        for s in sets:
            logger.info(
                "Set details: {}".format(helpers.set_details(s)))

        if logger.getEffectiveLevel() is logging.DEBUG:
            # view a couple of images to confirm that the class lables are
            # correct
            for s in sets:
                for i in range(5):
                    random_index = randrange(0, len(s[0]))
                    plot_qimage_grayscale(
                        s[0][random_index], s[1][random_index])

        # train_set, valid_set, test_set format: tuple(input, target)
        # input is an numpy.ndarray of 2 dimensions (a matrix)
        # witch row's correspond to an example. target is a
        # numpy.ndarray of 1 dimensions (vector)) that have the same length as
        # the number of rows in the input. It should give the target
        # target to the example with the same index in the input.

        # move the data to shared theano datasets
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                    (test_set_x, test_set_y)]

        self.datasets = datasets

        self.compute_training_parameters()
