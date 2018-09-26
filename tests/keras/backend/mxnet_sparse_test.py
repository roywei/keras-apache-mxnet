import warnings

import numpy as np
import pytest
import scipy.sparse as sparse
from keras import backend as K
from numpy.testing import assert_allclose

BACKEND = None

try:
    from keras.backend import mxnet_backend as KMX

    BACKEND = KMX
except ImportError:
    KMX = None
    warnings.warn('Could not import the MXNet backend')

pytestmark = pytest.mark.skipif(K.backend() != 'mxnet',
                                reason='Testing sparse support only for MXNet backend')


class TestMXNetSparse(object):
    def generate_test_sparse_matrix(self):
        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))
        return x_sparse

    def test_is_sparse(self):
        test_sparse_matrix = self.generate_test_sparse_matrix()
        test_var = K.variable(test_sparse_matrix)

        assert K.is_sparse(test_var)

    def test_is_sparse_matrix(self):
        test_sparse_matrix = self.generate_test_sparse_matrix()
        assert K.is_sparse(test_sparse_matrix)

    def test_to_dense(self):
        test_sparse_matrix = self.generate_test_sparse_matrix()
        test_var = K.variable(test_sparse_matrix)

        assert_allclose(K.to_dense(test_var), test_sparse_matrix.toarray())

    def test_to_dense_matrix(self):
        test_sparse_matrix = self.generate_test_sparse_matrix()

        assert_allclose(K.to_dense(test_sparse_matrix), test_sparse_matrix.toarray())

    def test_sparse_sum(self):
        test_sparse_matrix = self.generate_test_sparse_matrix()

        x_dense = test_sparse_matrix.toarray()
        test_var = K.variable(test_sparse_matrix)
        dense_var = K.variable(x_dense)

        k_s = K.eval(K.sum(test_var, axis=0))
        k_d = K.eval(K.sum(dense_var, axis=0))

        assert K.is_sparse(test_var)
        assert k_s.shape == k_d.shape
        assert_allclose(k_s, k_d, atol=1e-05)

    def test_sparse_mean(self):
        test_sparse_matrix = self.generate_test_sparse_matrix()

        x_dense = test_sparse_matrix.toarray()
        test_var = K.variable(test_sparse_matrix)
        dense_var = K.variable(x_dense)

        k_s = K.eval(K.mean(test_var, axis=0))
        k_d = K.eval(K.mean(dense_var, axis=0))

        assert K.is_sparse(test_var)
        assert k_s.shape == k_d.shape
        assert_allclose(k_s, k_d, atol=1e-05)

    def test_sparse_mean_axis_none(self):
        test_sparse_matrix = self.generate_test_sparse_matrix()

        x_dense = test_sparse_matrix.toarray()
        test_var = K.variable(test_sparse_matrix)
        dense_var = K.variable(x_dense)

        k_s = K.eval(K.mean(test_var))
        k_d = K.eval(K.mean(dense_var))

        assert K.is_sparse(test_var)
        assert k_s.shape == k_d.shape
        assert_allclose(k_s, k_d, atol=1e-05)

    def test_sparse_dot(self):
        x_sparse_1 = self.generate_test_sparse_matrix()
        x_dense_1 = x_sparse_1.toarray()

        W = np.random.random((5, 4))

        t_W = K.variable(W)
        k_s = K.eval(K.dot(K.variable(x_sparse_1), t_W))
        k_d = K.eval(K.dot(K.variable(x_dense_1), t_W))

        assert k_s.shape == k_d.shape
        assert_allclose(k_s, k_d, atol=1e-05)

    @pytest.mark.skip('diabled temporary until sparse_concat is fixed ')
    def test_sparse_concat(self):
        x_sparse_1 = self.generate_test_sparse_matrix()
        x_sparse_2 = self.generate_test_sparse_matrix()

        assert K.is_sparse(K.variable(x_sparse_1))
        assert K.is_sparse(K.variable(x_sparse_2))
        x_dense_1 = x_sparse_1.toarray()
        x_dense_2 = x_sparse_2.toarray()

        k_s = K.concatenate(tensors=[K.variable(x_sparse_1), K.variable(x_sparse_2)], axis=0)
        assert K.is_sparse(k_s)

        k_s_d = K.eval(k_s)

        # mx.sym.sparse.concat only supported for axis=0
        k_d = K.eval(K.concatenate(tensors=[K.variable(x_dense_1), K.variable(x_dense_2)], axis=0))

        assert k_s_d.shape == k_d.shape
        assert_allclose(k_s_d, k_d, atol=1e-05)

    def test_sparse_concat_partial_dense(self):
        x_sparse_1 = self.generate_test_sparse_matrix()
        x_sparse_2 = self.generate_test_sparse_matrix()

        assert K.is_sparse(K.variable(x_sparse_1))
        x_dense_1 = x_sparse_1.toarray()
        x_dense_2 = x_sparse_2.toarray()

        k_s = K.concatenate(tensors=[K.variable(x_sparse_1), K.variable(x_dense_2)], axis=0)
        assert not(K.is_sparse(k_s))

        k_s_d = K.eval(k_s)

        # mx.sym.sparse.concat only supported for axis=0
        k_d = K.eval(K.concatenate(tensors=[K.variable(x_dense_1), K.variable(x_dense_2)], axis=0))

        assert k_s_d.shape == k_d.shape
        assert_allclose(k_s_d, k_d, atol=1e-05)

    def test_sparse_concat_axis_non_zero(self):
        x_sparse_1 = self.generate_test_sparse_matrix()
        x_sparse_2 = self.generate_test_sparse_matrix()

        assert K.is_sparse(K.variable(x_sparse_1))
        x_dense_1 = x_sparse_1.toarray()
        x_dense_2 = x_sparse_2.toarray()

        k_s = K.concatenate(tensors=[K.variable(x_sparse_1), K.variable(x_dense_2)])
        assert not (K.is_sparse(k_s))

        k_s_d = K.eval(k_s)

        # mx.sym.sparse.concat only supported for axis=0
        k_d = K.eval(K.concatenate(tensors=[K.variable(x_dense_1), K.variable(x_dense_2)]))

        assert k_s_d.shape == k_d.shape
        assert_allclose(k_s_d, k_d, atol=1e-05)


if __name__ == '__main__':
    pytest.main([__file__])
