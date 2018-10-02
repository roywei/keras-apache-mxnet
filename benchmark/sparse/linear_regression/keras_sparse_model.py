"""
Linear Regression model with sparse synthetic data for Keras
"""
from __future__ import print_function

import time

from keras import Model, Sequential
from keras.layers import Dense, Input, Activation
from keras.optimizers import SGD
from keras import backend as K

from keras.utils import  multi_gpu_model
def _validate_backend():
    if K.backend() != 'mxnet' and K.backend() != 'tensorflow':
        raise NotImplementedError('This benchmark script only supports MXNet and TensorFlow backend')


def run_benchmark(train_data, train_label, eval_data, eval_label, batch_size, epochs, gpus):

    _validate_backend()

    inputs = Input(batch_shape=(None, train_data.shape[1]), dtype='float32', sparse=True)
    predictions = Dense(units=1, activation='linear', kernel_initializer='normal')(inputs)
    model = Model(inputs=inputs, outputs=predictions)
    model.summary()

    sgd = SGD(lr=0.1, momentum=0.9)
    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)
    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
    start = time.time()
    model.fit(train_data,
              train_label,
              epochs=epochs,
              batch_size=batch_size, verbose=1)

    print("Keras Benchmark Results")
    print("Dataset: Synthetic Sparse Data")
    print("Backend: ", K.backend().capitalize())
    print("Batch Size: ", batch_size)
    print("Total Time: ", time.time() - start)

    mse = model.evaluate(eval_data, eval_label, verbose=0, batch_size=batch_size)

    print("Achieved {0:.6f} validation MSE".format(mse[0]))
    print(model.evaluate(eval_data, eval_label, verbose=1, batch_size=batch_size))
