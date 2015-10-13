__author__ = 'Ahmad'


import os
import sys
import timeit
import cPickle

import numpy

import theano
import theano.tensor as T

from logisticsRegression_SGD import load_data
from Features import mapping_generator


class LogisticRegression(object):


    def __init__(self, input, n_in, n_out):

        
        # initialize theta = (W,b) with 0s; W gets the shape (n_in, n_out),
        # while b is a vector of n_out elements, making theta a vector of
        # n_in*n_out + n_out elements
        self.theta = theano.shared(
            value=numpy.zeros(
                n_in * n_out + n_out,
                dtype=theano.config.floatX
            ),
            name='theta',
            borrow=True
        )

        # W is represented by the fisr n_in*n_out elements of theta
        self.W = self.theta[0:n_in * n_out].reshape((n_in, n_out))
        # b is the rest (last n_out elements)
        self.b = self.theta[n_in * n_out:n_in * n_out + n_out]

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def cg_optimization_mnist(n_epochs=500, free917LR='lex_free917LR.pkl'):

    #############
    # LOAD DATA #
    #############
    datasets = load_data(free917LR)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    #lex:100, comp:10
    batch_size = 100   # size of the minibatch

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    #lex: 2035, 1718, comp:3579 , 175:
    n_in = 2035  # number of input units
    n_out = 1718  # number of output units

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    minibatch_offset = T.lscalar()  # offset to the start of a [mini]batch
    x = T.matrix()   # the data is presented as rasterized images
    y = T.ivector()  # the labels are presented as 1D vector of
                     # [int] labels

    # construct the logistic regression class
    classifier = LogisticRegression(input=x, n_in=2035, n_out=1718)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y).mean()

    # compile a theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        [minibatch_offset],
        classifier.errors(y),
        givens={
            x: test_set_x[minibatch_offset:minibatch_offset + batch_size],
            y: test_set_y[minibatch_offset:minibatch_offset + batch_size]
        },
        name="test"
    )

    validate_model = theano.function(
        [minibatch_offset],
        classifier.errors(y),
        givens={
            x: valid_set_x[minibatch_offset: minibatch_offset + batch_size],
            y: valid_set_y[minibatch_offset: minibatch_offset + batch_size]
        },
        name="validate"
    )

    #  compile a theano function that returns the cost of a minibatch
    batch_cost = theano.function(
        [minibatch_offset],
        cost,
        givens={
            x: train_set_x[minibatch_offset: minibatch_offset + batch_size],
            y: train_set_y[minibatch_offset: minibatch_offset + batch_size]
        },
        name="batch_cost"
    )

    # compile a theano function that returns the gradient of the minibatch
    # with respect to theta
    batch_grad = theano.function(
        [minibatch_offset],
        T.grad(cost, classifier.theta),
        givens={
            x: train_set_x[minibatch_offset: minibatch_offset + batch_size],
            y: train_set_y[minibatch_offset: minibatch_offset + batch_size]
        },
        name="batch_grad"
    )

    # creates a function that computes the average cost on the training set
    def train_fn(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        train_losses = [batch_cost(i * batch_size)
                        for i in xrange(n_train_batches)]
        return numpy.mean(train_losses)
    # creates a function that computes the average gradient of cost with
    # respect to theta
    def train_fn_grad(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        grad = batch_grad(0)
        for i in xrange(1, n_train_batches):
            grad += batch_grad(i * batch_size)
        return grad / n_train_batches

    validation_scores = [numpy.inf, 0]

    # creates the validation function
    def callback(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        #compute the validation loss
        validation_losses = [validate_model(i * batch_size)
                             for i in xrange(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses)
        print('validation error %f %%' % (this_validation_loss * 100.,))

        # check if it is better then best validation score got until now
        if this_validation_loss < validation_scores[0]:
            # if so, replace the old one, and compute the score on the
            # testing dataset
            with open('best_model.pkl', 'w') as f: cPickle.dump(classifier, f)
            validation_scores[0] = this_validation_loss
            test_losses = [test_model(i * batch_size)
                           for i in xrange(n_test_batches)]
            validation_scores[1] = numpy.mean(test_losses)


    ###############
    # TRAIN MODEL #
    ###############

    # using scipy conjugate gradient optimizer
    import scipy.optimize

    print ("Optimizing using scipy.optimize.fmin_cg...")
    start_time = timeit.default_timer()
    best_w_b = scipy.optimize.fmin_cg(
        f=train_fn,
        x0=numpy.zeros((n_in + 1) * n_out, dtype=x.dtype),
        fprime=train_fn_grad,
        callback=callback,
        disp=0,
        maxiter=n_epochs
    )

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, with '
            'test performance %f %%'
        )
        % (validation_scores[0] * 100., validation_scores[1] * 100.)
    )

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +

                          ' ran for %.1fs' % ((end_time - start_time)))
def predict():

    #Back to real world:
    functions, entities, chunks, vocs, uni_entity_mapping = mapping_generator()

    def symtransfer(in_list, in_array):
        indexes=[x for x in range(len(in_list)) if in_array[x]==1]
        out=''
        for i in indexes:
            out+=' '+in_list[i]
        return out

    # load the saved model
    classifier = cPickle.load(open('best_model.pkl'))
    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    dataset='lex_free917LR.pkl'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x)
    print ("Predicted values for the examples in test set:")

    for i in range(test_set_x.shape[0]):
        print symtransfer(vocs,test_set_x[i]),functions[predicted_values[i]]


if __name__ == '__main__':
    cg_optimization_mnist()
    predict()
