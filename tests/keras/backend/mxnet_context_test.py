import warnings

import pytest
import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import mxnet as mx

BACKEND = None

try:
    from keras.backend import mxnet_backend as KMX
    BACKEND = KMX
except ImportError:
    KMX = None
    warnings.warn('Could not import the MXNet backend')

pytestmark = pytest.mark.skipif(K.backend() != 'mxnet',
                                reason='Testing MXNet context supports only for MXNet backend')


class TestMXNetContext(object):
    batch_size = 128
    num_classes = 10
    epochs = 1

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)[:500]
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = keras.utils.to_categorical(y_train[:500], num_classes)
    gpus = mx.test_utils.list_gpus()
    if len(gpus) > 0:
        context = 'gpu(%d)' % gpus[-1]
    else:
        context = 'cpu'

    def _get_model(self):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(784,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model

    def test_context_model(self):
        model = Sequential(context=self.context)
        model.add(Dense(512, activation='relu', input_shape=(784,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

        model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs)

    def test_context_compile(self):
        model = self._get_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'],
                      context=self.context)

        model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs)

    def test_set_mxnet_context(self):
        model = self._get_model()
        model.set_mxnet_context(self.context)
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

        model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs)

if __name__ == '__main__':
    pytest.main([__file__])
