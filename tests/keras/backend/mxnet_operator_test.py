import pytest

import numpy as np
from keras import backend as K
from keras import Model
from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.models import load_model

pytestmark = pytest.mark.skipif(K.backend() != 'mxnet',
                                reason='Testing MXNet context supports only for MXNet backend')


class TestMXNetOperator(object):

    def test_batchnorm_freeze(self):
        data = np.array([
            [i, i ** 2, i + 2] for i in range(10)
        ])
        label = np.array([
            [i % 2] for i in range(10)
        ])
        x = Input(shape=(3,), name='x')
        f = Dense(10, name='h1')(x)
        f = BatchNormalization(name='bn1')(f)
        f = Activation('relu', name='r1')(f)
        y = Dense(1, name='y')(f)

        model = Model(inputs=[x], outputs=[y])
        model.compile(loss='binary_crossentropy', optimizer='sgd')
        model.fit(data, label, batch_size=5, epochs=10, verbose=1)
        free_layers = {'y'}
        for layer in model.layers:
            if layer.name not in free_layers:
                layer.trainable = False
        path = '/tmp/test_model.hdf5'
        model.save(path)

        loaded = load_model(path)
        loaded.compile(loss='binary_crossentropy', optimizer='sgd')

        loaded.fit(data, label, batch_size=5, epochs=10, verbose=1)
        for l1, l2 in zip(model.layers, loaded.layers):
            l1_weights = l1.get_weights()
            l2_weights = l2.get_weights()
            check = True
            for idx in range(len(l1_weights)):
                check &= np.allclose(l1_weights[idx], l2_weights[idx])
            # only the last layers have different weights
            if l1.name != 'y':
                assert check is True


if __name__ == '__main__':
    pytest.main([__file__])
