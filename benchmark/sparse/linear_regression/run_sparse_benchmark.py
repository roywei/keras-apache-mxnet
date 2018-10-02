"""
Prepare data for running benchmark on sparse linear regression model
"""
from __future__ import print_function

import argparse
import time

import keras_sparse_model
import mxnet as mx
import mxnet_sparse_model
from scipy import sparse

from keras import backend as K
from keras.utils.data_utils import prepare_sliced_sparse_data

def invoke_benchmark(batch_size, epochs, gpus):
    mx.random.seed(7)
    feature_dimension = 10000
    train_data = mx.test_utils.rand_ndarray((100000, feature_dimension), 'csr', 0.01)
    target_weight = mx.nd.arange(1, feature_dimension + 1).reshape((feature_dimension, 1))
    train_label = mx.nd.dot(train_data, target_weight)
    eval_data = train_data
    eval_label = mx.nd.dot(eval_data, target_weight)

    train_data = prepare_sliced_sparse_data(train_data, batch_size)
    train_label = prepare_sliced_sparse_data(train_label, batch_size)
    eval_data = prepare_sliced_sparse_data(eval_data, batch_size)
    eval_label = prepare_sliced_sparse_data(eval_label, batch_size)

    t_data = sparse.csr_matrix(train_data.asnumpy())
    e_data = sparse.csr_matrix(eval_data.asnumpy())
    t_label = train_label.asnumpy()
    e_label = eval_label.asnumpy()

    print("Running Keras benchmark script on sparse data")
    print("Using Backend: ", K.backend())
    keras_sparse_model.run_benchmark(train_data=t_data,
                                     train_label=t_label,
                                     eval_data=e_data,
                                     eval_label=e_label,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     gpus=gpus)

    print("Running MXNet benchmark script on sparse data")
    mxnet_sparse_model.run_benchmark(train_data=train_data,
                                     train_label=train_label,
                                     eval_data=eval_data,
                                     eval_label=eval_label,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     gpus=gpus)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=128,
                        help="Batch of data to be processed for training")
    parser.add_argument("--epochs", default=10,
                        help="Number of epochs to train the model on. Set epochs>=1000 for the best results")
    parser.add_argument("--gpus", default=0,
                        help="Number of epochs to train the model on. Set epochs>=1000 for the best results")
    args = parser.parse_args()

    invoke_benchmark(int(args.batch), int(args.epochs), int(args.gpus))
