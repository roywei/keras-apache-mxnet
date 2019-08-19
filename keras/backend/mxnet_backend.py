# -*- coding: utf-8 -*-
from __future__ import print_function

import warnings

import mxnet as mx
import numpy as np
from subprocess import CalledProcessError
from numbers import Number
from functools import wraps
from collections import defaultdict

from .common import floatx, epsilon, image_data_format

_UID_PREFIXES = defaultdict(int)
_LEARNING_PHASE = 1
_MODEL = None
_REENTRY = False
NAME_SCOPE_STACK = []

_CURRENT_SCOPE_CTX = None

py_all = all


class name_scope(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        global NAME_SCOPE_STACK
        NAME_SCOPE_STACK.append(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        NAME_SCOPE_STACK.pop()


def is_reentry():
    return _REENTRY


def set_reentry(value):
    global _REENTRY
    assert type(value) == bool, 'Please set to a boolean value.'
    _REENTRY = value


def set_model(model):
    global _MODEL
    _MODEL = model


def clear_session():
    global _MODEL
    reset_uids()
    _MODEL = None


def learning_phase():
    return _LEARNING_PHASE  # 0 = test, 1 = train


def set_learning_phase(value):
    global _LEARNING_PHASE
    if value not in {0, 1}:
        raise ValueError('MXNet Backend: Expected learning phase to be '
                         '0 or 1.')
    _LEARNING_PHASE = value


def keras_mxnet_symbol(func):
    """Decorator function used with all Keras API implementation. This
    decorator helps to establish the symbolic graph for the result MXNet symbol
    generated in the operator function.

    # Arguments
        func: Decorated function reference.
    """

    @wraps(func)
    def func_wrapper(*args, **kwargs):
        global _REENTRY
        reset = False
        try:
            if _REENTRY:
                train_symbol = func(*args, **kwargs)
                test_symbol = train_symbol
            else:
                _REENTRY = True
                reset = True
                actual_learning_phase = learning_phase()
                # Create Train Symbol
                set_learning_phase(1)
                train_symbol = func(*args, **kwargs)
                # Create Test Symbol
                set_learning_phase(0)
                test_symbol = func(*args, **kwargs)
                set_learning_phase(actual_learning_phase)
                assert type(train_symbol) == type(test_symbol)

            train_symbols = []
            test_symbols = []
            if isinstance(train_symbol, tuple):
                train_symbols = list(train_symbol)
                test_symbols = list(test_symbol)
            if isinstance(train_symbol, KerasSymbol):
                train_symbols = [train_symbol]
                test_symbols = [test_symbol]
            assert len(train_symbols) == len(test_symbols)
            for train_r, test_r in zip(train_symbols, test_symbols):
                assert type(train_r) == type(test_r)
                if isinstance(train_r, KerasSymbol):
                    train_r = [train_r]
                    test_r = [test_r]
                for train_i, test_i in zip(train_r, test_r):
                    if isinstance(train_i, KerasSymbol):
                        for arg in list(args) + list(kwargs.values()) + list(test_i.get_neighbor()):
                            if isinstance(arg, (list, tuple)):
                                for t in arg:
                                    train_i.add_neighbor(t)
                            else:
                                train_i.add_neighbor(arg)
                        if reset:
                            assert isinstance(train_i._train_sym, mx.sym.Symbol)
                            assert isinstance(test_i._pred_sym, mx.sym.Symbol)
                            assert train_i._name == test_i._name
                            train_i._pred_sym = test_i._pred_sym
                            assert train_i._train_sym is not None and train_i._pred_sym is not None
                    else:
                        assert (train_i == test_i) is True

            if reset:
                _REENTRY = False
            return train_symbol
        finally:
            if reset:
                _REENTRY = False

    return func_wrapper


# VARIABLE MANIPULATION
def cast_to_floatx(x):
    """Cast a Numpy array to the default Keras float type.

    # Arguments
        x: Numpy array.

    # Returns
        The same Numpy array, cast to its new type.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.floatx()
        'float32'
        >>> arr = numpy.array([1.0, 2.0], dtype='float64')
        >>> arr.dtype
        dtype('float64')
        >>> new_arr = K.cast_to_floatx(arr)
        >>> new_arr
        array([ 1.,  2.], dtype=float32)
        >>> new_arr.dtype
        dtype('float32')
    ```
    """
    x = np.asarray(x, dtype=floatx())
    if x.shape:
        return x
    else:
        return x.tolist()


def is_sparse(tensor):
    if hasattr(tensor, 'tocoo'):
        return True
    elif isinstance(tensor, KerasSymbol):
        if tensor.get_stype() == 'csr':
            return True
    return False


def to_dense(tensor):
    """Converts a sparse tensor into a dense tensor
    and returns it.

    # Arguments
        tensor: A tensor instance (potentially sparse).

    # Returns
        A dense tensor.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> b = K.placeholder((2, 2), sparse=True)
        >>> print(K.is_sparse(b))
        True
        >>> c = K.to_dense(b)
        >>> print(K.is_sparse(c))
        False
    ```
    """
    if hasattr(tensor, 'tocoo'):
        return tensor.toarray()
    elif isinstance(tensor, mx.sym.Symbol):
        return tensor.stype('default')
    elif isinstance(tensor, KerasSymbol):
        return eval(tensor)
    else:
        return tensor


def variable(value, dtype=None, name=None, constraint=None, sparse_weight=False):
    """Instantiates a variable and returns it.

    # Arguments
        value: Numpy array, initial value of the tensor.
        dtype: Tensor type.
        name: Optional name string for the tensor.
        constraint: Optional projection function to be
            applied to the variable after an optimizer update.
        sparse_weight: Parameter used for MXNet backend to use sparse weight for training.
    # Returns
        A variable instance (with Keras metadata included).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val, dtype='float64', name='example_var')
        >>> K.dtype(kvar)
        'float64'
        >>> print(kvar)
        example_var
        >>> kvar.eval()
        array([[ 1.,  2.],
               [ 3.,  4.]])
    ```
    """
    if constraint:
        warnings.warn('MXNet backend does not support constraints. Keyword '
                      'arguments such as `kernel_constraint` and '
                      '`bias_constraint`', stacklevel=2)
    if dtype is None:
        dtype = floatx()

    is_vector = False
    if hasattr(value, "shape") and value.shape == (1,):
        is_vector = True
    if isinstance(value, Number):
        value = np.array([value])
    if isinstance(value, KerasSymbol):
        value = eval(value)

    if hasattr(value, 'tocoo'):
        v = mx.nd.sparse.csr_matrix((value.data, value.indices, value.indptr), shape=value.shape)
        name = _prepare_name(name, 'variable')
        ret = _keras_variable(name, v.shape, v.dtype, 'csr', is_vector)
        ret._keras_shape = value.shape
        ret._uses_learning_phase = False
        ret.bind(v)
        return ret
    elif sparse_weight:
        v = mx.nd.array(value).tostype('row_sparse')
        name = _prepare_name(name, 'variable')
        ret = _keras_variable(name, v.shape, v.dtype, 'row_sparse', is_vector)
        ret._keras_shape = value.shape
        ret._uses_learning_phase = False
        ret.bind(v)
        return ret

    # MXNet backend do not support scalars
    if isinstance(value, np.ndarray) and len(value.shape) == 0:
        raise ValueError('MXNet Backend: Scalars are not supported. Provided value for variable '
                         '- ', value)

    dtype = _convert_string_dtype(dtype)
    name = _prepare_name(name, 'variable')
    ndarray = mx.nd.array(value, dtype=dtype)
    ret = _keras_variable(name, ndarray.shape, ndarray.dtype, 'default', is_vector)
    ret.bind(ndarray)

    if isinstance(value, np.ndarray):
        ret._keras_shape = tuple([d if d != 0 else None for d in value.shape])
    elif hasattr(value, 'shape'):
        ret._keras_shape = tuple([d if d != 0 else None for d in map(int, value.shape)])
    ret._uses_learning_phase = False

    return ret


@keras_mxnet_symbol
def constant(value, dtype=None, shape=None, name=None):
    """Creates a constant tensor.

    # Arguments
        value: A constant value (or list)
        dtype: The type of the elements of the resulting tensor.
        shape: Optional dimensions of resulting tensor.
        name: Optional name for the tensor.

    # Returns
        A Constant Tensor.
    """
    if dtype is None:
        dtype = floatx()

    if type(value) is np.ndarray:
        mx_ndarray = mx.nd.array(value, dtype=dtype)
    elif type(value) is list:
        mx_ndarray = mx.nd.array(value, dtype=dtype)
    elif shape is None:
        mx_ndarray = mx.nd.array(value, dtype=dtype)
    else:
        shape = tuple([0 if dim is None else dim for dim in shape])
        np_ndarray = np.ndarray(shape, dtype=dtype)
        np_ndarray.fill(value)
        mx_ndarray = mx.nd.array(np_ndarray)

    # MXNet does not support Scalars. Shape of a Scalar Tensor with MXNet is
    # (1, ) instead of (). Return is as MXNet NDArray instance as this is
    # useful in K.eval() function to return as is (1, )
    if shape == (1,):
        return mx_ndarray
    else:
        name = _prepare_name(name, 'constant')
        const_var = _keras_variable(name=name, dtype=dtype, shape=mx_ndarray.shape)
        const_var.bind(mx_ndarray)
        return const_var


def is_keras_tensor(x):
    """Returns whether `x` is a Keras tensor.

    A "Keras tensor" is a tensor that was returned by a Keras layer,
    (`Layer` class) or by `Input`.

    # Arguments
        x: A candidate tensor.

    # Returns
        A boolean: Whether the argument is a Keras tensor.

    # Raises
        ValueError: In case `x` is not a symbolic tensor.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> from keras.layers import Input, Dense
        >>> np_var = numpy.array([1, 2])
        >>> K.is_keras_tensor(np_var) # A numpy array is not a symbolic tensor.
        ValueError
        >>> k_var = tf.placeholder('float32', shape=(1,1))
        >>> K.is_keras_tensor(k_var) # A variable indirectly created outside of keras is not a Keras tensor.
        False
        >>> keras_var = K.variable(np_var)
        >>> K.is_keras_tensor(keras_var)  # A variable created with the keras backend is not a Keras tensor.
        False
        >>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
        >>> K.is_keras_tensor(keras_placeholder)  # A placeholder is not a Keras tensor.
        False
        >>> keras_input = Input([10])
        >>> K.is_keras_tensor(keras_input) # An Input is a Keras tensor.
        True
        >>> keras_layer_output = Dense(10)(keras_input)
        >>> K.is_keras_tensor(keras_layer_output) # Any Keras layer output is a Keras tensor.
        True
    ```
    """
    if not isinstance(x, KerasSymbol):
        raise ValueError('MXNet Backend: Unexpectedly found an instance of type `' +
                         str(type(x)) + '`.''Expected a symbolic tensor instance.')
    return hasattr(x, '_keras_history')


def is_tensor(x):
    return isinstance(x, KerasSymbol) or isinstance(x, mx.sym.Symbol) or isinstance(x, mx.nd.NDArray)


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    """Instantiates a placeholder tensor and returns it.

    # Arguments
        shape: Shape of the placeholder
            (integer tuple, may include `None` entries).
        ndim: Number of axes of the tensor.
            At least one of {`shape`, `ndim`} must be specified.
            If both are specified, `shape` is used.
        dtype: Placeholder type.
        sparse: Boolean, whether the placeholder should have a sparse type.
        name: Optional name string for the placeholder.

    # Returns
        Tensor instance (with Keras metadata included).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> input_ph = K.placeholder(shape=(2, 4, 5))
        >>> input_ph._keras_shape
        (2, 4, 5)
        >>> input_ph
        placeholder1:[tensor=False dtype=float32]
    ```
    """
    if dtype is None:
        dtype = floatx()

    dtype = _convert_string_dtype(dtype)
    if shape is None and ndim is None:
        raise ValueError('MXNet Backend: Specify either a shape or ndim value.')
    name = _prepare_name(name, 'placeholder')

    if shape:
        shape = tuple([0 if dim is None else dim for dim in shape])
    else:
        shape = tuple([0 for _ in range(ndim)])

    if sparse:
        sym = _keras_variable(name, shape=shape, dtype=dtype, stype='csr')
        sym._keras_shape = tuple([d if d != 0 else None for d in shape])
        sym._mxnet_placeholder = True
        sym._uses_learning_phase = False
        return sym

    sym = _keras_variable(name, shape=shape, dtype=dtype, stype='default')
    sym._keras_shape = tuple([d if d != 0 else None for d in shape])
    sym._mxnet_placeholder = True
    sym._uses_learning_phase = False
    return sym


def is_placeholder(x):
    """Returns whether `x` is a placeholder.

    # Arguments
        x: A candidate placeholder.

    # Returns
        Boolean.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> input_ph = K.placeholder(shape=(2,4,5))
        >>> K.is_placeholder(input_ph)
        True
    ```
    """
    return hasattr(x, '_mxnet_placeholder') and x._mxnet_placeholder


def shape(x):
    """Returns the symbolic shape of a tensor or variable.

    # Arguments
        x: A tensor or variable.

    # Returns
        A symbolic shape (which is itself a tensor).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> K.shape(kvar)
        (2,3)
        >>> K.shape(inputs)
        (2,4,5)
    ```
    """
    if isinstance(x, KerasSymbol):
        return x.shape
    else:
        return None


def int_shape(x):
    """Returns the shape tensor or variable as a tuple of int or None entries.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tuple of integers (or None entries).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> K.int_shape(inputs)
        (2, 4, 5)
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.int_shape(kvar)
        (2, 2)
    ```
    """
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    try:
        if type(x.shape) is tuple:
            return x.shape
        else:
            return tuple(x.shape.as_list())
    except ValueError:
        return None


def ndim(x):
    """Returns the number of axes in a tensor, as an integer.

    # Arguments
        x: Tensor or variable.

    # Returns
        Integer (scalar), number of axes.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.ndim(inputs)
        3
        >>> K.ndim(kvar)
        2
    ```
    """
    shape = x.shape
    if shape is not None:
        return len(shape)
    return None


def dtype(x):
    """Returns the dtype of a Keras tensor or variable, as a string.

    # Arguments
        x: Tensor or variable.

    # Returns
        String, dtype of `x`.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> K.dtype(K.placeholder(shape=(2,4,5)))
        'float32'
        >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float32'))
        'float32'
        >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float64'))
        'float64'
        # Keras variable
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]))
        >>> K.dtype(kvar)
        'float32'
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
        >>> K.dtype(kvar)
        'float32'
    ```
    """
    return x.dtype


def eval(x):
    """Evaluates the value of a variable.

    # Arguments
        x: A variable.

    # Returns
        A Numpy array.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
        >>> K.eval(kvar)
        array([[ 1.,  2.],
               [ 3.,  4.]], dtype=float32)
    ```
    """
    if isinstance(x, KerasSymbol):
        if x.tensor is not None:
            if x.name in x.get_bind_values() and _MODEL is not None:
                _MODEL._sync_weights()
            ret = x.eval().asnumpy()
        else:
            ret = _forward_pass(x)[0].asnumpy()

        # If the Tensor shape is (1, ) and does not have attribute "_is_vector", then, it is considered to be scalar.
        # Return the value.
        if ret.shape == (1,) and not hasattr(x, '_is_vector'):
            ret = ret[0]

        return ret
    elif isinstance(x, mx.nd.NDArray):
        return x.asnumpy()
    else:
        return x


def zeros(shape, dtype=None, name=None):
    """Instantiates an all-zeros variable and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable
        dtype: String, data type of returned Keras variable
        name: String, name of returned Keras variable

    # Returns
        A variable (including Keras metadata), filled with `0.0`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.zeros((3,4))
        >>> K.eval(kvar)
        array([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]], dtype=float32)
    ```
    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    shape = tuple([0 if dim is None else dim for dim in shape])
    value = mx.nd.zeros(shape, dtype=dtype)
    name = _prepare_name(name, 'zeroinit')
    kvar = _keras_variable(name, value.shape, value.dtype)
    kvar.bind(value)
    return kvar


def ones(shape, dtype=None, name=None):
    """Instantiates an all-ones tensor variable and returns it.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable.
        dtype: String, data type of returned Keras variable.
        name: String, name of returned Keras variable.

    # Returns
        A Keras variable, filled with `1.0`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.ones((3,4))
        >>> K.eval(kvar)
        array([[ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.]], dtype=float32)
    ```
    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    shape = tuple([0 if dim is None else dim for dim in shape])
    value = mx.nd.ones(shape, dtype=dtype)
    name = _prepare_name(name, 'oneinit')
    kvar = _keras_variable(name=name, shape=shape, dtype=dtype)
    kvar.bind(value)
    return kvar


def eye(size, dtype=None, name=None):
    """Instantiate an identity matrix and returns it.

    # Arguments
        size: Integer, number of rows/columns.
        dtype: String, data type of returned Keras variable.
        name: String, name of returned Keras variable.

    # Returns
        A Keras variable, an identity matrix.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.eye(3)
        >>> K.eval(kvar)
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]], dtype=float32)
    ```

    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    value = mx.nd.array(np.eye(size, dtype=dtype))
    name = _prepare_name(name, 'eyeinit')
    kvar = _keras_variable(name=name, shape=size, dtype=dtype)
    kvar.bind(value)
    return kvar


@keras_mxnet_symbol
def zeros_like(x, dtype=None, name=None):
    """Instantiates an all-zeros variable of the same shape as another tensor.

    # Arguments
        x: Keras variable or Keras tensor.
        dtype: String, dtype of returned Keras variable.
             None uses the dtype of x.
        name: String, name for the variable to create.

    # Returns
        A Keras variable with the shape of x filled with zeros.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.variable(np.random.random((2,3)))
        >>> kvar_zeros = K.zeros_like(kvar)
        >>> K.eval(kvar_zeros)
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]], dtype=float32)
    ```
    """
    return KerasSymbol(mx.sym.zeros_like(data=x.symbol, name=_prepare_name(name, 'zeroslikeinit')))


def ones_like(x, dtype=None, name=None):
    """Instantiates an all-ones variable of the same shape as another tensor.

    # Arguments
        x: Keras variable or tensor.
        dtype: String, dtype of returned Keras variable.
             None uses the dtype of x.
        name: String, name for the variable to create.

    # Returns
        A Keras variable with the shape of x filled with ones.

    # Example
    ```python
        >>> from keras import backend as K
        >>> kvar = K.variable(np.random.random((2,3)))
        >>> kvar_ones = K.ones_like(kvar)
        >>> K.eval(kvar_ones)
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
    ```
    """
    return KerasSymbol(mx.sym.ones_like(data=x.symbol, name=_prepare_name(name, 'oneslikeinit')))


def identity(x):
    """Returns a tensor with the same content as the input tensor.

    # Arguments
        x: The input tensor.

    # Returns
        A tensor of the same shape, type and content.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
        >>> kvar_copy = K.identity(kvar)
        >>> K.eval(kvar)
        array([[ 1.,  2.],
               [ 3.,  4.]], dtype=float32)
        >>> K.eval(kvar_copy)
        array([[ 1.,  2.],
               [ 3.,  4.]], dtype=float32)
    ```
    """
    name = _prepare_name(None, 'identityinit')
    dtype = x.dtype
    x_value = eval(x)
    mx_shape = tuple([0 if x is None else x for x in x.shape])
    mx_value = mx.nd.array(x_value, dtype=dtype)
    k_var = _keras_variable(name=name, dtype=dtype, shape=mx_shape)
    k_var.bind(mx_value)
    return k_var


def random_uniform_variable(shape, low, high, dtype=None,
                            name=None, seed=None):
    """Instantiates a variable with values drawn from a uniform distribution.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable.
        low: Float, lower boundary of the output interval.
        high: Float, upper boundary of the output interval.
        dtype: String, dtype of returned Keras variable.
        name: String, name of returned Keras variable.
        seed: Integer, random seed.

    # Returns
        A Keras variable, filled with drawn samples.

    # Example
    ```python
        >>> rand_var = K.random_uniform_variable(shape=(3,3), low=1, high=3)
        >>> K.eval(rand_var)
        array([[ 2.09762716,  2.18568921,  2.43037868],
               [ 2.6885314 ,  2.20552683,  2.71589136],
               [ 2.0897665 ,  2.69450331,  1.84730959]], dtype=float32)
        >>> rand_var1 = K.random_uniform_variable(shape=(3,3), low=1, high=3, seed=128)
        >>> rand_var2 = K.random_uniform_variable(shape=(3,3), low=1, high=3, seed=128)
        >>> K.eval(rand_var1)
        array([[ 1.07625926,  1.60461187,  1.30567968],
               [ 2.43757105,  2.77134657,  2.16382241],
               [ 1.7636764 ,  2.51851654,  1.96760654]], dtype=float32)
        >>> K.eval(rand_var2)
        array([[ 1.07625926,  1.60461187,  1.30567968],
               [ 2.43757105,  2.77134657,  2.16382241],
               [ 1.7636764 ,  2.51851654,  1.96760654]], dtype=float32)
    ```
    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    name = _prepare_name(name, 'randomuniform')
    if seed:
        mx.random.seed(seed)
    value = mx.random.uniform(low=low, high=high, dtype='float32', shape=shape)
    if dtype != np.float32:
        value = mx.nd.Cast(value, dtype=dtype)
    k_var = _keras_variable(name=name, shape=shape, dtype=dtype)
    k_var.bind(value)
    return k_var


def random_normal_variable(shape, mean, scale, dtype=None,
                           name=None, seed=None):
    """Instantiates a variable with values drawn from a normal distribution.

    # Arguments
        shape: Tuple of integers, shape of returned Keras variable.
        mean: Float, mean of the normal distribution.
        scale: Float, standard deviation of the normal distribution.
        dtype: String, dtype of returned Keras variable.
        name: String, name of returned Keras variable.
        seed: Integer, random seed.

    # Returns
        A Keras variable, filled with drawn samples.

    # Example
    ```python
        >>> rand_var = K.random_normal_variable(shape=(3,3), mean=0, scale=1)
        >>> K.eval(rand_var)
        array([[ 1.16307867,  2.21220636,  0.48380461],
               [ 0.7740038 ,  0.29956347,  1.04344034],
               [ 0.15302546,  1.18392551, -1.16881478]], dtype=float32)
        >>> rand_var1 = K.random_normal_variable(shape=(3,3), mean=0, scale=1, seed=128)
        >>> rand_var2 = K.random_normal_variable(shape=(3,3), mean=0, scale=1, seed=128)
        >>> K.eval(rand_var1)
        array([[-0.75213492,  0.47400656,  0.95352972],
               [ 0.20251541, -0.62203991,  1.36481571],
               [-0.08511394, -1.4962182 , -0.20014545]], dtype=float32)
        >>> K.eval(rand_var2)
        array([[-0.75213492,  0.47400656,  0.95352972],
               [ 0.20251541, -0.62203991,  1.36481571],
               [-0.08511394, -1.4962182 , -0.20014545]], dtype=float32)
    ```
    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    name = _prepare_name(name, 'randomnormal')
    if seed:
        mx.random.seed(seed)
    value = mx.random.normal(loc=mean, scale=scale, dtype='float32', shape=shape)
    if dtype != np.float32:
        value = mx.nd.Cast(value, dtype=dtype)
    k_var = _keras_variable(name=name, shape=shape, dtype=dtype)
    k_var.bind(value)
    return k_var


def count_params(x):
    """Returns the number of scalars in a Keras variable.

    # Arguments
        x: Keras variable.

    # Returns
        Integer, the number of scalars in `x`.

    # Example
    ```python
        >>> k_var = K.zeros((2,3))
        >>> K.count_params(k_var)
        6
        >>> K.eval(k_var)
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]], dtype=float32)
    ```
    """
    shape = tuple([0 if x is None else x for x in x.shape])
    return np.prod([shape[i] for i in range(len(shape))])


@keras_mxnet_symbol
def cast(x, dtype):
    """Casts a tensor to a different dtype and returns it.

    You can cast a Keras variable but it still returns a Keras tensor.

    # Arguments
        x: Keras tensor (or variable).
        dtype: String, either (`'float16'`, `'float32'`, or `'float64'`).

    # Returns
        Keras tensor with dtype `dtype`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> input = K.placeholder((2, 3), dtype='float32')
        >>> input
        placeholder1:[tensor=True dtype=float32]
        # It doesn't work in-place as below.
        >>> K.cast(input, dtype='float16')
        cast0:[tensor=True dtype=float16]
        >>> input
        placeholder1:[tensor=True dtype=float32]
        # you need to assign it.
        >>> input = K.cast(input, dtype='float16')
        >>> input
        cast2:[tensor=True dtype=float16]
    ```
    """
    if isinstance(x, KerasSymbol):
        return KerasSymbol(
            mx.sym.Cast(data=x.symbol, dtype=dtype))
    elif hasattr(x, 'astype'):
        return x.astype(dtype)
    else:
        raise TypeError('MXNet Backend: The input is not valid for cast operation.')


# UPDATES OPS
def update(x, new_x):
    """Update the value of `x` to `new_x`.

    # Arguments
        x: A `Variable`.
        new_x: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    raise NotImplementedError('MXNet Backend: Update operations are not supported yet.')


def update_add(x, increment):
    """Update the value of `x` by adding `increment`.

    # Arguments
        x: A `Variable`.
        increment: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    raise NotImplementedError('MXNet Backend: Update operations are not supported yet.')


def update_sub(x, decrement):
    """Update the value of `x` by subtracting `decrement`.

    # Arguments
        x: A `Variable`.
        decrement: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    raise NotImplementedError('MXNet Backend: Update operations are not supported yet.')


@keras_mxnet_symbol
def moving_average_update(x, value, momentum):
    """Compute the moving average of a variable.

    # Arguments
        x: A `Variable`.
        value: A tensor with the same shape as `x`.
        momentum: The moving average momentum.

    # Returns
        An operation to update the variable.
    """
    return KerasSymbol(x.symbol * momentum + value * (1. - momentum))


# LINEAR ALGEBRA
@keras_mxnet_symbol
def dot(x, y):
    """Multiplies 2 tensors (and/or variables) and returns a *tensor*.

    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor, dot product of `x` and `y`.

    # Examples
    ```python
        # dot product between tensors
        >>> x = K.placeholder(shape=(2, 3))
        >>> y = K.placeholder(shape=(3, 4))
        >>> xy = K.dot(x, y)
        >>> xy
        dot0:[tensor=True dtype=float32]
        >>> xy.shape
        (2, 4)
    ```

    ```python
        # dot product between tensors
        >>> x = K.placeholder(shape=(32, 28, 3))
        >>> y = K.placeholder(shape=(3, 4))
        >>> xy = K.dot(x, y)
        >>> xy
        dot0:[tensor=True dtype=float32]
        >>> xy.shape
        (32, 28, 4)
    ```

    ```python
        # Theano-like behavior example
        >>> x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
        >>> y = K.ones((4, 3, 5))
        >>> xy = K.dot(x, y)
        >>> K.int_shape(xy)
        (2, 4, 5)
    ```
    """
    if is_sparse(x):
        if ndim(y) > 2:
            # Output of this is sparse
            return KerasSymbol(mx.sym.sparse.dot(x.symbol, y.symbol, transpose_b=True))
        else:
            # Output of this is dense
            return KerasSymbol(mx.sym.sparse.dot(x.symbol, y.symbol))

    if ndim(y) > 2:
        axis = list(range(ndim(y)))
        axis = [axis.pop(-2)] + axis
        y = mx.sym.transpose(y.symbol, axes=axis)
    else:
        y = y.symbol
    return KerasSymbol(mx.sym.dot(lhs=x.symbol, rhs=y))


@keras_mxnet_symbol
def batch_dot(x, y, axes=None):
    """Batchwise dot product.

    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.

    # Arguments
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.

    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.

    # Examples
        Assume `x = [[1, 2], [3, 4]]` and `y = [[5, 6], [7, 8]]`
        `batch_dot(x, y, axes=1) = [[17, 53]]` which is the main diagonal
        of `x.dot(y.T)`, although we never have to calculate the off-diagonal
        elements.

        Shape inference:
        Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.
        If `axes` is (1, 2), to find the output shape of resultant tensor,
            loop through each dimension in `x`'s shape and `y`'s shape:

        * `x.shape[0]` : 100 : append to output shape
        * `x.shape[1]` : 20 : do not append to output shape,
            dimension 1 of `x` has been summed over. (`dot_axes[0]` = 1)
        * `y.shape[0]` : 100 : do not append to output shape,
            always ignore first dimension of `y`
        * `y.shape[1]` : 30 : append to output shape
        * `y.shape[2]` : 20 : do not append to output shape,
            dimension 2 of `y` has been summed over. (`dot_axes[1]` = 2)
        `output_shape` = `(100, 30)`

    ```python
        >>> x_batch = K.ones(shape=(32, 20, 1))
        >>> y_batch = K.ones(shape=(32, 30, 20))
        >>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
        >>> K.int_shape(xy_batch_dot)
        (32, 1, 30)
    ```
    """
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = KerasSymbol(mx.sym.reshape(y.symbol, shape=shape(y) + (1,) * diff))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = KerasSymbol(mx.sym.reshape(x.symbol, shape=shape(x) + (1,) * diff))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        # MXNet supports only 3D. Expand_dims to make it 3D. At the end squeeze it back.
        x = expand_dims(x, axis=1)
        y = expand_dims(y, axis=2)
        diff = 1
        if axes[0] == axes[1]:
            out = KerasSymbol(mx.sym.batch_dot(lhs=x.symbol, rhs=y.symbol))
        else:
            out = KerasSymbol(mx.sym.batch_dot(lhs=x.symbol, rhs=y.symbol,
                                               transpose_a=True))
    else:
        if axes is not None:
            trans_x = False if axes[0] == ndim(x) - 1 else True
            trans_y = True if axes[1] == ndim(y) - 1 else False
        else:
            trans_x = False
            trans_y = False
        out = KerasSymbol(mx.sym.linalg_gemm2(x.symbol, y.symbol, transpose_a=trans_x,
                                              transpose_b=trans_y))
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = squeeze(out, idx)
    if ndim(out) == 1:
        out = expand_dims(out, 1)
    return out


@keras_mxnet_symbol
def transpose(x):
    """Transposes a tensor and returns it.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.

    # Examples
    ```python
        >>> var = K.variable([[1, 2, 3], [4, 5, 6]])
        >>> K.eval(var)
        array([[ 1.,  2.,  3.],
               [ 4.,  5.,  6.]], dtype=float32)
        >>> var_transposed = K.transpose(var)
        >>> K.eval(var_transposed)
        array([[ 1.,  4.],
               [ 2.,  5.],
               [ 3.,  6.]], dtype=float32)
    ```

    ```python
        >>> inputs = K.placeholder((2, 3))
        >>> inputs
        placeholder1:[tensor=True dtype=float32]
        >>> inputs.shape
        (2,3)
        >>> input_transposed = K.transpose(inputs)
        >>> input_transposed
        transpose1:[tensor=True dtype=float32]
        >>> input_transposed.shape
        (3,2)
    ```
    """
    return KerasSymbol(
        mx.sym.transpose(data=x.symbol))


@keras_mxnet_symbol
def gather(reference, indices):
    """Retrieves the elements of indices `indices` in the tensor `reference`.

    # Arguments
        reference: A tensor.
        indices: An integer tensor of indices.

    # Returns
        A tensor of same type as `reference`.
    """
    indices = mx.sym.Cast(indices.symbol, dtype=reference.dtype)
    return KerasSymbol(mx.sym.take(reference.symbol, indices))


@keras_mxnet_symbol
def embedding(data, weight, input_dim, output_dim, sparse_grad=False):
    # check if inputs are KerasSymbol
    if isinstance(data, KerasSymbol):
        data = data.symbol
    if isinstance(weight, KerasSymbol):
        weight = weight.symbol
    if sparse_grad:
        # Refer https://mxnet.incubator.apache.org/api/python/symbol/sparse.html#mxnet.symbol.sparse.Embedding
        return KerasSymbol(mx.sym.Embedding(data, weight=weight, input_dim=input_dim, output_dim=output_dim,
                                            sparse_grad=True))
    return KerasSymbol(mx.sym.Embedding(data, weight=weight, input_dim=input_dim, output_dim=output_dim))


# ELEMENT-WISE OPERATIONS
@keras_mxnet_symbol
def max(x, axis=None, keepdims=False):
    """Maximum value in a tensor.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to find maximum values.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with maximum values of `x`.
    """
    axis = _normalize_axis(axis, ndim(x))
    return KerasSymbol(mx.sym.max(data=x.symbol, axis=axis, keepdims=keepdims))


@keras_mxnet_symbol
def min(x, axis=None, keepdims=False):
    """Minimum value in a tensor.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to find minimum values.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with miminum values of `x`.
    """
    axis = _normalize_axis(axis, ndim(x))
    return KerasSymbol(mx.sym.min(data=x.symbol, axis=axis, keepdims=keepdims))


@keras_mxnet_symbol
def sum(x, axis=None, keepdims=False):
    """Sum of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to sum over.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with sum of `x`.
    """
    axis = _normalize_axis(axis, ndim(x))
    return KerasSymbol(mx.sym.sum(data=x.symbol, axis=axis, keepdims=keepdims))


@keras_mxnet_symbol
def prod(x, axis=None, keepdims=False):
    """Multiplies the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the product.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the product of elements of `x`.
    """
    axis = _normalize_axis(axis, ndim(x))
    return KerasSymbol(mx.sym.prod(data=x.symbol, axis=axis, keepdims=keepdims))


def cumsum(x, axis=0):
    """Cumulative sum of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the sum.

    # Returns
        A tensor of the cumulative sum of values of `x` along `axis`.
    """
    raise NotImplementedError('MXNet Backend: cumsum operator is not supported yet.')


def cumprod(x, axis=0):
    """Cumulative product of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the product.

    # Returns
        A tensor of the cumulative product of values of `x` along `axis`.
    """
    raise NotImplementedError('MXNet Backend: cumprod operator is not supported yet.')


def _mxnet_variance(x, axis=None, keepdims=False):
    mean_input = mx.sym.mean(data=x, axis=axis, keepdims=True)
    centered_input = mx.sym.broadcast_minus(lhs=x, rhs=mean_input)
    v = mx.sym.mean(data=(centered_input ** 2), axis=axis, keepdims=keepdims)
    return v


@keras_mxnet_symbol
def var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    axis = _normalize_axis(axis, ndim(x))
    if isinstance(x, KerasSymbol):
        x = x.symbol
    v = _mxnet_variance(x, axis=axis, keepdims=keepdims)
    return KerasSymbol(v)


@keras_mxnet_symbol
def std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    v = var(x, axis=axis, keepdims=keepdims)
    ret = mx.sym.sqrt(data=v.symbol)
    return KerasSymbol(ret)


@keras_mxnet_symbol
def mean(x, axis=None, keepdims=False):
    """Mean of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: A list of integer. Axes to compute the mean.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1 for each entry in `axis`. If `keep_dims` is `True`,
            the reduced dimensions are retained with length 1.

    # Returns
        A tensor with the mean of elements of `x`.
    """
    if axis == []:
        return x
    axis = _normalize_axis(axis, ndim(x))
    if dtype(x) == 'uint8':
        x = cast(x, floatx())
    if axis is not None:
        ret = mx.sym.mean(data=x.symbol, axis=axis, keepdims=keepdims)
    else:
        ret = mx.sym.mean(data=x.symbol, keepdims=keepdims)
    return KerasSymbol(ret)


@keras_mxnet_symbol
def any(x, axis=None, keepdims=False):
    """Bitwise reduction (logical OR).

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A uint8 tensor (0s and 1s).
    """

    axis = _normalize_axis(axis, ndim(x))
    if isinstance(x, KerasSymbol):
        x = x.symbol
    non_zero = (x != 0)
    var_cast = mx.sym.Cast(data=non_zero, dtype=np.int32)
    var_sum = mx.sym.sum_axis(data=var_cast, axis=axis, keepdims=keepdims)

    return KerasSymbol(var_sum > 0)


@keras_mxnet_symbol
def all(x, axis=None, keepdims=False):
    """Bitwise reduction (logical AND).

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A uint8 tensor (0s and 1s).
    """
    axis = _normalize_axis(axis, ndim(x))
    if isinstance(x, KerasSymbol):
        x = x.symbol
    var_abs = mx.sym.abs(data=x)
    var_min = mx.sym.min_axis(data=var_abs, axis=axis, keepdims=keepdims)
    return KerasSymbol(var_min > 0)


@keras_mxnet_symbol
def argmax(x, axis=-1):
    """Returns the index of the maximum value along an axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.

    # Returns
        A tensor.
    """
    axis = _normalize_axis(axis, ndim(x))
    ret = mx.sym.argmax(data=x.symbol, axis=axis)
    return KerasSymbol(ret)


@keras_mxnet_symbol
def argmin(x, axis=-1):
    """Returns the index of the minimum value along an axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.

    # Returns
        A tensor.
    """
    axis = _normalize_axis(axis, ndim(x))
    ret = mx.sym.argmin(data=x.symbol, axis=axis)
    return KerasSymbol(ret)


@keras_mxnet_symbol
def square(x):
    """Element-wise square.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.square(data=x.symbol))


@keras_mxnet_symbol
def abs(x):
    """Element-wise absolute value.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.abs(data=x.symbol))


@keras_mxnet_symbol
def sqrt(x):
    """Element-wise square root.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    ret = mx.sym.Activation(data=x.symbol, act_type='relu')
    ret = mx.sym.sqrt(data=ret)
    return KerasSymbol(ret)


@keras_mxnet_symbol
def exp(x):
    """Element-wise exponential.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.exp(data=x.symbol))


@keras_mxnet_symbol
def log(x):
    """Element-wise log.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.log(data=x.symbol))


@keras_mxnet_symbol
def logsumexp(x, axis=None, keepdims=False):
    """Computes log(sum(exp(elements across dimensions of a tensor))).

    This function is more numerically stable than log(sum(exp(x))).
    It avoids overflows caused by taking the exp of large inputs and
    underflows caused by taking the log of small inputs.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to reduce over.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`, the reduced dimension is
            retained with length 1.

    # Returns
        The reduced tensor.
    """
    raise NotImplementedError('MXNet Backend: logsumexp operator is not supported yet.')


@keras_mxnet_symbol
def round(x):
    """Element-wise rounding to the closest integer.

    In case of tie, the rounding mode used is "half to even".

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.round(data=x.symbol))


@keras_mxnet_symbol
def sign(x):
    """Element-wise sign.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.sign(data=x.symbol))


@keras_mxnet_symbol
def pow(x, a):
    """Element-wise exponentiation.

    # Arguments
        x: Tensor or variable.
        a: Python integer.

    # Returns
        A tensor.
    """
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(a, KerasSymbol):
        a = a.symbol
    return KerasSymbol(mx.sym.pow(base=x, exp=a))


@keras_mxnet_symbol
def clip(x, min_value, max_value):
    """Element-wise value clipping.

    # Arguments
        x: Tensor or variable.
        min_value: Python float or integer.
        max_value: Python float or integer.

    # Returns
        A tensor.
    """
    if max_value is not None and max_value < min_value:
        max_value = min_value
    if max_value is None:
        max_value = np.inf
    min_value = np.float32(min_value)
    max_value = np.nan_to_num(np.float32(max_value))
    return KerasSymbol(mx.sym.clip(data=x.symbol, a_min=min_value, a_max=max_value))


@keras_mxnet_symbol
def equal(x, y):
    """Element-wise equality between two tensors.

    For all element-wise comparison operators:
    use broadcasting to do element-wise comparison if both x & y are MXNet symbol
    use native comparison operators for scalar
    use numpy operators if both x & y are numbers or numpy arrays

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.

    # Raise
        TypeError: if inputs are not valid.
    """
    scalar = False
    if isinstance(x, KerasSymbol):
        x = x.symbol
        scalar = True
    if isinstance(y, KerasSymbol):
        y = y.symbol
        scalar = True
    if isinstance(x, mx.sym.Symbol) and isinstance(y, mx.sym.Symbol):
        out = KerasSymbol(mx.sym.Cast(mx.sym.broadcast_equal(lhs=x, rhs=y), dtype='uint8'))
    elif scalar:
        out = KerasSymbol(mx.sym.Cast(x == y, dtype='uint8'))
    else:
        try:
            out = np.equal(x, y)
        except:
            raise TypeError('MXNet Backend: The inputs are not valid for equal operation.')
    return out


@keras_mxnet_symbol
def not_equal(x, y):
    """Element-wise inequality between two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.

    # Raise
        TypeError: if inputs are not valid.
    """
    scalar = False
    if isinstance(x, KerasSymbol):
        x = x.symbol
        scalar = True
    if isinstance(y, KerasSymbol):
        y = y.symbol
        scalar = True
    if isinstance(x, mx.sym.Symbol) and isinstance(y, mx.sym.Symbol):
        out = KerasSymbol(mx.sym.Cast(mx.sym.broadcast_not_equal(lhs=x, rhs=y), dtype='uint8'))
    elif scalar:
        out = KerasSymbol(mx.sym.Cast(x != y, dtype='uint8'))
    else:
        try:
            out = np.not_equal(x, y)
        except:
            raise TypeError('MXNet Backend: The inputs are not valid for not_equal operation.')
    return out


@keras_mxnet_symbol
def greater(x, y):
    """Element-wise truth value of (x > y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.

    # Raise
        TypeError: if inputs are not valid.
    """
    scalar = False
    if isinstance(x, KerasSymbol):
        x = x.symbol
        scalar = True
    if isinstance(y, KerasSymbol):
        y = y.symbol
        scalar = True
    if isinstance(x, mx.sym.Symbol) and isinstance(y, mx.sym.Symbol):
        out = KerasSymbol(mx.sym.Cast(mx.sym.broadcast_greater(lhs=x, rhs=y), dtype='uint8'))
    elif scalar:
        out = KerasSymbol(mx.sym.Cast(x > y, dtype='uint8'))
    else:
        try:
            out = np.greater(x, y)
        except:
            raise TypeError('MXNet Backend: The inputs are not valid for greater operation.')
    return out


@keras_mxnet_symbol
def greater_equal(x, y):
    """Element-wise truth value of (x >= y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.

    # Raise
        TypeError: if inputs are not valid.
    """
    scalar = False
    if isinstance(x, KerasSymbol):
        x = x.symbol
        scalar = True
    if isinstance(y, KerasSymbol):
        y = y.symbol
        scalar = True
    if isinstance(x, mx.sym.Symbol) and isinstance(y, mx.sym.Symbol):
        out = KerasSymbol(mx.sym.Cast(mx.sym.broadcast_greater_equal(lhs=x, rhs=y), dtype='uint8'))
    elif scalar:
        out = KerasSymbol(mx.sym.Cast(x >= y, dtype='uint8'))
    else:
        try:
            out = np.greater_equal(x, y)
        except:
            raise TypeError('MXNet Backend: The inputs are not valid for greater_equal operation.')
    return out


@keras_mxnet_symbol
def less(x, y):
    """Element-wise truth value of (x < y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.

    # Raise
        TypeError: if inputs are not valid.
    """
    scalar = False
    if isinstance(x, KerasSymbol):
        x = x.symbol
        scalar = True
    if isinstance(y, KerasSymbol):
        y = y.symbol
        scalar = True
    if isinstance(x, mx.sym.Symbol) and isinstance(y, mx.sym.Symbol):
        out = KerasSymbol(mx.sym.Cast(mx.sym.broadcast_lesser(lhs=x, rhs=y), dtype='uint8'))
    elif scalar:
        out = KerasSymbol(mx.sym.Cast(x < y, dtype='uint8'))
    else:
        try:
            out = np.less(x, y)
        except:
            raise TypeError('MXNet Backend: The inputs are not valid for less operation.')
    return out


@keras_mxnet_symbol
def less_equal(x, y):
    """Element-wise truth value of (x <= y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.

    # Raise
        TypeError: if inputs are not valid.
    """
    scalar = False
    if isinstance(x, KerasSymbol):
        x = x.symbol
        scalar = True
    if isinstance(y, KerasSymbol):
        y = y.symbol
        scalar = True
    if isinstance(x, mx.sym.Symbol) and isinstance(y, mx.sym.Symbol):
        out = KerasSymbol(mx.sym.Cast(mx.sym.broadcast_lesser_equal(lhs=x, rhs=y), dtype='uint8'))
    elif scalar:
        out = KerasSymbol(mx.sym.Cast(x <= y, dtype='uint8'))
    else:
        try:
            out = np.less(x, y)
        except:
            raise TypeError('MXNet Backend: The inputs are not valid for less_equal operation.')
    return out


@keras_mxnet_symbol
def maximum(x, y):
    """Element-wise maximum of two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor.
    """
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(y, KerasSymbol):
        y = y.symbol
    return KerasSymbol(mx.sym.maximum(left=x, right=y))


@keras_mxnet_symbol
def minimum(x, y):
    """Element-wise minimum of two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor.
    """
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(y, KerasSymbol):
        y = y.symbol
    return KerasSymbol(mx.sym.minimum(left=x, right=y))


@keras_mxnet_symbol
def sin(x):
    """Computes sin of x element-wise.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.sin(data=x.symbol))


@keras_mxnet_symbol
def cos(x):
    """Computes cos of x element-wise.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.cos(data=x.symbol))


@keras_mxnet_symbol
def normalize_batch_in_training(x, gamma, beta,
                                reduction_axes, epsilon=1e-3):
    """Computes mean and std for batch then apply batch_normalization on batch.

    # Arguments
        x: Input tensor or variable.
        gamma: Tensor by which to scale the input.
        beta: Tensor with which to center the input.
        reduction_axes: iterable of integers,
            axes over which to normalize.
        epsilon: Fuzz factor.

    # Returns
        A tuple length of 3, `(normalized_tensor, mean, variance)`.
    """
    original_x = x
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(beta, KerasSymbol):
        beta = beta.symbol
    if isinstance(gamma, KerasSymbol):
        gamma = gamma.symbol

    mean = mx.sym.mean(data=x, axis=reduction_axes, keepdims=False)
    var = _mxnet_variance(x, axis=reduction_axes, keepdims=False)

    if sorted(reduction_axes) == list(range(ndim(original_x)))[:-1]:
        normed = batch_normalization(x, mean, var, beta, gamma, epsilon)
    else:
        # need broadcasting
        target_shape = []
        for axis in range(ndim(original_x)):
            if axis in reduction_axes:
                target_shape.append(1)
            else:
                target_shape.append(original_x.shape[axis])
        target_shape = tuple(target_shape)

        broadcast_mean = mx.sym.Reshape(data=mean, shape=target_shape)
        broadcast_var = mx.sym.Reshape(data=var, shape=target_shape)
        broadcast_gamma = mx.sym.Reshape(data=gamma, shape=target_shape)
        broadcast_beta = mx.sym.Reshape(data=beta, shape=target_shape)
        normed = batch_normalization(x, broadcast_mean, broadcast_var,
                                     broadcast_beta, broadcast_gamma,
                                     epsilon)

    return normed, KerasSymbol(mean), KerasSymbol(var)


@keras_mxnet_symbol
def batch_normalization(x, mean, var, beta, gamma, epsilon=1e-3):
    """Applies batch normalization on x given mean, var, beta and gamma.

    I.e. returns:
    `output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`

    # Arguments
        x: Input tensor or variable.
        mean: Mean of batch.
        var: Variance of batch.
        beta: Tensor with which to center the input.
        gamma: Tensor by which to scale the input.
        epsilon: Fuzz factor.

    # Returns
        A tensor.
    """
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(mean, KerasSymbol):
        mean = mean.symbol
    if isinstance(var, KerasSymbol):
        var = var.symbol
    if isinstance(beta, KerasSymbol):
        beta = beta.symbol
    if isinstance(gamma, KerasSymbol):
        gamma = gamma.symbol

    mean = mx.sym.stop_gradient(mean)
    var = mx.sym.stop_gradient(var)

    # gradient explode when learning gamma and beta at together.
    gamma = mx.sym.stop_gradient(gamma)

    std = mx.sym.sqrt(data=(var + epsilon))

    x = mx.sym.broadcast_minus(x, mean)
    x = mx.sym.broadcast_div(x, std)
    x = mx.sym.broadcast_mul(x, gamma)
    x = mx.sym.broadcast_plus(x, beta)
    return KerasSymbol(x)


@keras_mxnet_symbol
def mxnet_batchnorm(x, gamma, beta, moving_mean, moving_var, momentum=0.9, axis=1, epsilon=1e-3):
    """Apply native  MXNet batch normalization on x with given moving_mean,
    moving_var, beta and gamma.

    # Arguments
        x: Input tensor or variable.
        gamma: Tensor by which to scale the input.
        beta: Tensor by which to center the input.
        moving_mean: Moving mean.
        moving_var: Moving variance.
        momentum: Moving average momentum. Defaults to 0.9
        axis: Axis along which Batchnorm is applied. Axis usually represent axis of 'channels'. MXNet follows
        'channels_first' hence, defaults to '1'.
        epsilon: Fuzz factor to avoid divide by zero.

    # Returns
        A tensor.
    """
    if isinstance(x, KerasSymbol):
        x = x.symbol
    if isinstance(moving_mean, KerasSymbol):
        moving_mean = moving_mean.symbol
    if isinstance(moving_var, KerasSymbol):
        moving_var = moving_var.symbol
    if isinstance(beta, KerasSymbol):
        beta = beta.symbol
    if isinstance(gamma, KerasSymbol):
        gamma = gamma.symbol

    if axis != 1:
        warnings.warn('MXNet Backend uses `channels_first` format. Axis for BatchNorm should ideally be `1`.'
                      'Provided - `' + str(axis) + '`. Performance can be significantly lower!', stacklevel=2)

    return KerasSymbol(
        mx.sym.BatchNorm(x, gamma, beta, moving_mean,
                         moving_var, momentum=momentum, axis=axis, eps=epsilon))


# SHAPE OPERATIONS
@keras_mxnet_symbol
def concatenate(tensors, axis=-1):
    """Concatenates a list of tensors alongside the specified axis.

    # Arguments
        tensors: list of tensors to concatenate.
        axis: concatenation axis.

    # Returns
        A tensor.

    Note:
    - MXNet supports sparse concat only for dim=0
    - https://mxnet.apache.org/api/python/symbol/sparse.html#mxnet.symbol.sparse.concat
    """
    if axis < 0:
        rank = ndim(tensors[0])
        if rank:
            axis %= rank
        else:
            axis = 0

    symbols = [t.symbol for t in tensors]

    return KerasSymbol(mx.sym.concat(*symbols, dim=axis))


@keras_mxnet_symbol
def reshape(x, shape):
    """Reshapes a tensor to the specified shape.

    # Arguments
        x: Tensor or variable.
        shape: Target shape tuple.

    # Returns
        A tensor.
    """
    shape = tuple([0 if dim is None else dim for dim in shape])
    return KerasSymbol(mx.sym.reshape(data=x.symbol, shape=shape))


@keras_mxnet_symbol
def permute_dimensions(x, pattern):
    """Permutes axes in a tensor.

    # Arguments
        x: Tensor or variable.
        pattern: A tuple of
            dimension indices, e.g. `(0, 2, 1)`.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.transpose(x.symbol, axes=pattern))


@keras_mxnet_symbol
def resize_images(x, height_factor, width_factor, data_format, interpolation='nearest'):
    """Resizes the images contained in a 4D tensor.

    # Arguments
        x: Tensor or variable to resize.
        height_factor: Positive integer.
        width_factor: Positive integer.
        data_format: string, `"channels_last"` or `"channels_first"`.
        interpolation: A string, one of `nearest` or `bilinear`.

    # Returns
        A tensor.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    x = x.symbol
    if data_format == 'channels_last':
        x = mx.sym.repeat(x, repeats=height_factor, axis=1)
        x = mx.sym.repeat(x, repeats=width_factor, axis=2)
    elif data_format == 'channels_first':
        x = mx.sym.repeat(x, repeats=height_factor, axis=2)
        x = mx.sym.repeat(x, repeats=width_factor, axis=3)
    else:
        raise ValueError('MXNET Backend: Data format is neither channels_first or channels_last')

    return KerasSymbol(x)


@keras_mxnet_symbol
def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
    """Resizes the volume contained in a 5D tensor.

    # Arguments
        x: Tensor or variable to resize.
        depth_factor: Positive integer.
        height_factor: Positive integer.
        width_factor: Positive integer.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A tensor.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    x = x.symbol
    if data_format == 'channels_last':
        x = mx.sym.repeat(x, repeats=depth_factor, axis=1)
        x = mx.sym.repeat(x, repeats=height_factor, axis=2)
        x = mx.sym.repeat(x, repeats=width_factor, axis=3)
    elif data_format == 'channels_first':
        x = mx.sym.repeat(x, repeats=depth_factor, axis=2)
        x = mx.sym.repeat(x, repeats=height_factor, axis=3)
        x = mx.sym.repeat(x, repeats=width_factor, axis=4)
    else:
        raise ValueError('MXNET Backend: Data format is neither channels_first or channels_last')

    return KerasSymbol(x)


@keras_mxnet_symbol
def repeat_elements(x, rep, axis):
    """Repeats the elements of a tensor along an axis, like `np.repeat`.

    If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
    will have shape `(s1, s2 * rep, s3)`.

    # Arguments
        x: Tensor or variable.
        rep: Python integer, number of times to repeat.
        axis: Axis along which to repeat.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.repeat(x.symbol, repeats=rep, axis=axis))


@keras_mxnet_symbol
def repeat(x, n):
    """Repeats a 2D tensor.

    if `x` has shape (samples, dim) and `n` is `2`,
    the output will have shape `(samples, 2, dim)`.

    # Arguments
        x: Tensor or variable.
        n: Python integer, number of times to repeat.

    # Returns
        A tensor.
    """
    x = mx.sym.expand_dims(x.symbol, axis=1)
    x = mx.sym.repeat(x, repeats=n, axis=1)
    return KerasSymbol(x)


@keras_mxnet_symbol
def arange(start, stop=None, step=1, dtype='int32'):
    """Creates a 1D tensor containing a sequence of integers.

    The function arguments use the same convention as
    Theano's arange: if only one argument is provided,
    it is in fact the "stop" argument.

    The default type of the returned tensor is `'int32'` to
    match TensorFlow's default.

    # Arguments
        start: Start value.
        stop: Stop value.
        step: Difference between two successive values.
        dtype: Integer dtype to use.

    # Returns
        An integer tensor.

    """
    dtype = np.dtype(dtype)
    return KerasSymbol(mx.sym.arange(start=start, stop=stop, step=step, dtype=dtype))


@keras_mxnet_symbol
def tile(x, n):
    """Creates a tensor by tiling `x` by `n`.

    # Arguments
        x: A tensor or variable
        n: A list of integer. The length must be the same as the number of
            dimensions in `x`.

    # Returns
        A tiled tensor.
    """
    return KerasSymbol(mx.sym.tile(x.symbol, reps=n))


@keras_mxnet_symbol
def flatten(x):
    """Flatten a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor, reshaped into 1-D
    """
    return KerasSymbol(mx.sym.Reshape(data=x.symbol, shape=(-1,)))


@keras_mxnet_symbol
def batch_flatten(x):
    """Turn a nD tensor into a 2D tensor with same 0th dimension.

    In other words, it flattens each data samples of a batch.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.Flatten(data=x.symbol))


@keras_mxnet_symbol
def expand_dims(x, axis=-1):
    """Adds a 1-sized dimension at index "axis".

    # Arguments
        x: A tensor or variable.
        axis: Position where to add a new axis.

    # Returns
        A tensor with expanded dimensions.
    """
    if axis < 0:
        axis %= len(x.shape) + 1
    if isinstance(x, KerasSymbol):
        x = x.symbol
        return KerasSymbol(mx.sym.expand_dims(x, axis=axis))


@keras_mxnet_symbol
def squeeze(x, axis):
    """Removes a 1-dimension from the tensor at index "axis".

    # Arguments
        x: A tensor or variable.
        axis: Axis to drop.

    # Returns
        A tensor with the same data as `x` but reduced dimensions.
    """
    shape = list(x.shape)
    assert shape.pop(axis) == 1, 'Can only squeeze size 1 dimension'

    if isinstance(x, KerasSymbol):
        x = x.symbol
        return KerasSymbol(mx.sym.Reshape(data=x, shape=tuple(shape)))


@keras_mxnet_symbol
def temporal_padding(x, padding=(1, 1)):
    """Pads the middle dimension of a 3D tensor.

    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 integers, how many zeros to
            add at the start and end of dim 1.

    # Returns
        A padded 3D tensor.
    """
    assert len(padding) == 2

    assert ndim(x) == 3

    # MXNet only supports padding for 4D and 5D tensor.
    # Reshaping to 4D, perform padding, reshape back to 3D.
    x_shape = tuple([0 if dim is None else dim for dim in x.shape])
    x_4d = mx.sym.Reshape(x.symbol, shape=(x_shape[0], 1, x_shape[1], x_shape[2]))
    x_4d_padded = mx.sym.pad(data=x_4d, mode='constant', constant_value=0, pad_width=(0, 0, 0, 0, padding[0],
                                                                                      padding[1], 0, 0,))
    x_3d_padded = mx.sym.Reshape(x_4d_padded, shape=(x_shape[0], x_shape[1] + padding[0] + padding[1],
                                                     x_shape[2]))
    return KerasSymbol(x_3d_padded)


@keras_mxnet_symbol
def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    """Pads the 2nd and 3rd dimensions of a 4D tensor.

    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 tuples, padding pattern.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A padded 4D tensor.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    assert ndim(x) == 4

    if data_format is None:
        data_format = image_data_format()

    _validate_data_format(data_format)

    # Pre process input for handling data_format - channels_first/channels_last.
    # MXNet requires input to be in channels_first.
    x = _preprocess_convnd_input(x, data_format)

    pattern = (0, 0, 0, 0, padding[0][0], padding[0][1], padding[1][0], padding[1][1])
    x = KerasSymbol(mx.sym.Pad(data=x.symbol, mode='constant', constant_value=0, pad_width=pattern))

    # Convert back to original data_format
    x = _postprocess_convnd_output(x, data_format)
    return x


@keras_mxnet_symbol
def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    """Pads 5D tensor with zeros along the depth, height, width dimensions.

    Pads these dimensions with respectively
    "padding[0]", "padding[1]" and "padding[2]" zeros left and right.

    For 'channels_last' data_format,
    the 2nd, 3rd and 4th dimension will be padded.
    For 'channels_first' data_format,
    the 3rd, 4th and 5th dimension will be padded.

    # Arguments
        x: Tensor or variable.
        padding: Tuple of 3 tuples, padding pattern.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A padded 5D tensor.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.

    """
    assert len(padding) == 3
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    assert len(padding[2]) == 2

    assert ndim(x) == 5

    if data_format is None:
        data_format = image_data_format()

    _validate_data_format(data_format)

    # Pre process input for handling data_format - channels_first/channels_last.
    # MXNet requires input to be in channels_first.
    x = _preprocess_convnd_input(x, data_format)

    pattern = (
        0, 0,
        0, 0,
        padding[0][0], padding[0][1],
        padding[1][0], padding[1][1],
        padding[2][0], padding[2][1]
    )
    x = KerasSymbol(mx.sym.Pad(data=x.symbol, mode='constant', constant_value=0, pad_width=pattern))
    # Convert back to original data_format
    x = _postprocess_convnd_output(x, data_format)
    return x


def stack(x, axis=0):
    """Stacks a list of rank `R` tensors into a rank `R+1` tensor.

    # Arguments
        x: List of tensors.
        axis: Axis along which to perform stacking.

    # Returns
        A tensor.
    """
    raise NotImplementedError('MXNet Backend: Stack operation is not supported yet.')


@keras_mxnet_symbol
def one_hot(indices, num_classes):
    """Computes the one-hot representation of an integer tensor.

    # Arguments
        indices: nD integer tensor of shape
            `(batch_size, dim1, dim2, ... dim(n-1))`
        num_classes: Integer, number of classes to consider.

    # Returns
        (n + 1)D one hot representation of the input
        with shape `(batch_size, dim1, dim2, ... dim(n-1), num_classes)`
    """
    return KerasSymbol(mx.symbol.one_hot(indices.symbol, depth=num_classes))


@keras_mxnet_symbol
def reverse(x, axes):
    """Reverse a tensor along the specified axes.

    # Arguments
        x: Tensor to reverse.
        axes: Integer or iterable of integers.
            Axes to reverse.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.symbol.reverse(data=x.symbol, axis=axes))


# VALUE MANIPULATION
def get_value(x):
    """Returns the value of a variable.

    # Arguments
        x: input variable.

    # Returns
        A Numpy array.
    """
    return eval(x)


def get_mxnet_module_arg_params(x):
    """
    given parameter name,
    get the updated value from mxnet module arg_params
    :param x: parameter values to  (e.g. kernels, bias)
    :return: updated parameter values
    """
    # retrieve from bind values first, which is up to date with
    # arg_params in mxnet module
    ret = eval(x)
    if isinstance(x, KerasSymbol):
        if x.tensor is not None:
            if x.name in x.get_bind_values() and _MODEL is not None:
                _MODEL._sync_weights()
                ret = x.get_bind_values()[x.name].asnumpy()
    return ret


def batch_get_value(ops):
    """Returns the value of more than one tensor variable.

    # Arguments
        ops: list of ops to run.

    # Returns
        A list of Numpy arrays.
    """
    return [get_mxnet_module_arg_params(op) for op in ops]


def set_value(x, value):
    """Sets the value of a variable, from a Numpy array.

    # Arguments
        x: Tensor to set to a new value.
        value: Value to set the tensor to, as a Numpy array
            (of the same shape).
    """
    if isinstance(value, Number):
        value = [value]
    x.bind(mx.nd.array(value))


def batch_set_value(tuples):
    """Sets the values of many tensor variables at once.

    # Arguments
        tuples: a list of tuples `(tensor, value)`.
            `value` should be a Numpy array.
    """
    for p, w in tuples:
        set_value(p, w)


def get_variable_shape(x):
    """Returns the shape of a variable.

    # Arguments
        x: A variable.

    # Returns
        A tuple of integers.
    """
    return x.shape


def print_tensor(x, message=''):
    """Prints `message` and the tensor value when evaluated.

    # Arguments
        x: Tensor to print.
        message: Message to print jointly with the tensor.

    # Returns
        The same tensor `x`, unchanged.
    """
    print(message, eval(x))


@keras_mxnet_symbol
def group(variables):
    var = [x if isinstance(x, mx.sym.Symbol) else x.symbol for x in variables]
    sym = mx.sym.Group(var)
    return KerasSymbol(sym)


@keras_mxnet_symbol
def make_loss(variables):
    sym = mx.sym.MakeLoss(variables.symbol)
    return KerasSymbol(sym)


# GRAPH MANIPULATION
class Function(object):
    def __init__(self, inputs, output, updates=[], **kwargs):
        self.output = output
        self.updates = updates
        if isinstance(inputs[-1], Number):
            self.is_train = inputs[-1]
            self.inputs = inputs[:-1]
        else:
            self.inputs = inputs
            self.is_train = learning_phase()

    def __call__(self, inputs):
        ret_outputs = []
        if isinstance(inputs[-1], Number):
            self.is_train = inputs[-1]
            inputs = inputs[:-1]
        for x in self.output:
            bind_values = dfs_get_bind_values(x)
            data = {k.name: v for k, v in zip(self.inputs, inputs)}
            data = dict(data, **bind_values)
            args = x.symbol.list_arguments()
            data_shapes = {k.name: v.shape for k, v in zip(self.inputs, inputs) if k.name in args}
            executor = x.symbol.simple_bind(mx.cpu(), grad_req='null', **data_shapes)
            for v in executor.arg_dict:
                if v in data:
                    executor.arg_dict[v][:] = data[v]
            outputs = executor.forward(is_train=self.is_train)
            ret_outputs.append(outputs[0].asnumpy())
        return ret_outputs


def function(inputs, outputs, updates=None, **kwargs):
    """Instantiates a Keras function.

    # Arguments
        inputs: List of placeholder tensors.
        outputs: List of output tensors.
        updates: List of update ops.
        **kwargs: Passed to `tf.Session.run`.

    # Returns
        Output values as Numpy arrays.

    # Raises
        ValueError: if invalid kwargs are passed in.
    """
    return Function(inputs, outputs, updates=updates, **kwargs)


def gradients(loss, variables):
    """Returns the gradients of `variables` w.r.t. `loss`.

    # Arguments
        loss: Scalar tensor to minimize.
        variables: List of variables.

    # Returns
        A gradients tensor.
    """
    raise NotImplementedError('MXNet Backend: Gradients operator is not supported yet.')


@keras_mxnet_symbol
def stop_gradient(variables):
    """Returns `variables` but with zero gradient w.r.t. every other variable.

    # Arguments
        variables: tensor or list of tensors to consider constant with respect
            to any other variable.

    # Returns
        A single tensor or a list of tensors (depending on the passed argument)
            that has constant gradient with respect to any other variable.
    """
    if isinstance(variables, KerasSymbol):
        return KerasSymbol(mx.sym.BlockGrad(variables.symbol))
    elif isinstance(variables, list):
        out = []
        for variable in variables:
            out.append(KerasSymbol(mx.sym.BlockGrad(variable.symbol)))
        return out
    else:
        raise ValueError('MXNet backend: Stop gradient requires tensor or '
                         'list of tensors, but, given {0}'.format(type(variables)))


# CONTROL FLOW
@keras_mxnet_symbol
def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None, cell=None, training=None):
    """Iterates over the time dimension of a tensor.

    # Arguments
        step_function: RNN step function.
            Parameters:
                inputs: tensor with shape `(samples, ...)` (no time dimension),
                    representing input for the batch of samples at a certain
                    time step.
                states: list of tensors.
            Returns:
                outputs: tensor with shape `(samples, output_dim)`
                    (no time dimension).
                new_states: list of tensors, same length and shapes
                    as 'states'. The first state in the list must be the
                    output tensor at the previous timestep.
        inputs: tensor of temporal data of shape `(samples, time, ...)`
            (at least 3D).
        initial_states: tensor with shape (samples, output_dim)
            (no time dimension),
            containing the initial values for the states used in
            the step function.
        go_backwards: boolean. If True, do the iteration over the time
            dimension in reverse order and return the reversed sequence.
        mask: binary tensor with shape `(samples, time, 1)`,
            with a zero for every element that is masked.
        constants: a list of constant values passed at each step.
        unroll: whether to unroll the RNN or to use a symbolic loop
        (`while_loop` or `scan` depending on backend).
        input_length: not relevant in the MXNet implementation.
            Must be specified if using unrolling with Theano.

    # Returns
        A tuple, `(last_output, outputs, new_states)`.

            last_output: the latest output of the rnn, of shape `(samples, ...)`
            outputs: tensor with shape `(samples, time, ...)` where each
                entry `outputs[s, t]` is the output of the step function
                at time `t` for sample `s`.
            new_states: list of tensors, latest states returned by
                the step function, of shape `(samples, ...)`.

    # Raises
        ValueError: if input dimension is less than 3.
        ValueError: if `unroll` is `True` but input timestep is not a fixed number.
        ValueError: if `mask` is provided (not `None`) but states is not provided
            (`len(states)` == 0).
    """
    dtype = inputs.dtype
    dshape = inputs.shape

    if len(dshape) < 3:
        raise ValueError('MXNet Backend: Input tensor should be at least 3-D')

    if constants is None:
        constants = []
    # Assume learning phase is a placeholder tensor.(F = test, T = train)
    # Some Keras layers (e.g. Dropout, BatchNormalization) behave differently at
    #  training time and testing time. You can tell whether a layer uses the
    # "learning phase" (train/test) by printing layer.uses_learning_phase, a
    # boolean: True if the layer has a different behavior in training mode and
    # test mode, False otherwise.
    global uses_learning_phase
    uses_learning_phase = False

    # for custom operations when K.rnn is directly called to operate
    # on tensors (mostly unit tests), no cell information is provided,
    # use unrolling by default
    if not unroll and cell is None:
        unroll = True
        warnings.warn('MXNet Backend: K.rnn() is called without RNN cell information, '
                      'using unrolling by default.')

    if unroll:
        # Split the inputs across time dimension and generate the list of inputs
        # with shape `(samples, ...)` (no time dimension)
        inputs = list(mx.sym.split(inputs.symbol, axis=1,
                                   squeeze_axis=1, num_outputs=dshape[1]))

        # Reverse the input sequence
        if go_backwards:
            inputs.reverse()

        states = initial_states
        outputs = []
        prev_output = None

        if mask is not None:
            if not states:
                raise ValueError('MXNet Backend: Initial states is not provided when masking is '
                                 'enabled.')
            if mask.dtype != dtype:
                mask = cast(mask, dtype)
            # Split the mask across time dimension and generate the list of masks
            # with shape `(samples, 1)` (no time dimension)
            masks = list(mx.sym.split(mask.symbol, axis=1,
                                      squeeze_axis=1, num_outputs=dshape[1]))
            # Reverse the mask sequence
            if go_backwards:
                masks.reverse()
        else:
            masks = [None for _ in inputs]

        # Iterate over a time step
        for inp, msk in zip(inputs, masks):
            last_output, new_states = step_function(KerasSymbol(inp),
                                                    states + constants)
            if getattr(last_output, '_uses_learning_phase', False):
                uses_learning_phase = True
            if msk is not None:
                new_states = [KerasSymbol(mx.sym.where(msk,
                                                       ns.symbol,
                                                       s.symbol))
                              for s, ns in zip(states, new_states)]
                # Initialize the output for first time step
                if prev_output is None:
                    prev_output = zeros_like(last_output)
                last_output = KerasSymbol(mx.sym.where(msk,
                                                       last_output.symbol,
                                                       prev_output.symbol))
                prev_output = last_output
            states = new_states
            # Expand the output dimension from `(samples, output_dim)` to
            # `(samples, 1, output_dim)` with middle axis as time dimension
            outputs.append(mx.sym.expand_dims(last_output.symbol, axis=1))
        # Concatenate the output across time dimension
        outputs = mx.sym.concat(*outputs, dim=1)
    else:
        if mx.__version__ < '1.3.1':
            raise NotImplementedError('unroll=False in RNN only works with MXNet 1.3.1 or newer, '
                                      'please upgrade to latest master using: pip install --upgrade mxnet')
        # defining step functions for each RNN cells, implementation taken from call functions
        # from each RNN cell class in keras.layers.recurrent

        def _simple_rnn_cell_step(data, states):
            # Refer to SimpleRNNCell's call function in keras.layers.recurrent
            inputs = data[0]
            mask = None
            if len(data) > 1:
                mask = data[1]
            prev_output = states[0]

            # dropout matrices for input units
            dp_mask = None
            # dropout matrices for recurrent units
            rec_dp_mask = None
            if 0 < cell.dropout < 1 and cell._dropout_mask is None:
                dp_mask = _generate_dropout_mask(
                    KerasSymbol(mx.sym.ones_like(inputs)),
                    cell.dropout,
                    training=training)
            if (0 < cell.recurrent_dropout < 1 and
                    cell._recurrent_dropout_mask is None):
                rec_dp_mask = _generate_dropout_mask(
                    KerasSymbol(mx.sym.ones_like(prev_output)),
                    cell.recurrent_dropout,
                    training=training)

            if dp_mask is not None:
                h = _dot_rnn(inputs * dp_mask.symbol, cell.kernel.symbol)
            else:
                h = _dot_rnn(inputs, cell.kernel.symbol)
            if cell.bias is not None:
                h = mx.sym.broadcast_add(h, cell.bias.symbol)

            if rec_dp_mask is not None:
                prev_output = prev_output + rec_dp_mask.symbol
            outputs = h + _dot_rnn(prev_output, cell.recurrent_kernel.symbol)
            if cell.activation is not None:
                outputs = cell.activation(KerasSymbol(outputs)).symbol
            if mask is not None:
                outputs = mx.sym.where(mask, outputs, prev_output)
            return outputs, [outputs]

        def _lstm_cell_step(data, states):
            # Refer to LSTMCell's call function in keras.layers.recurrent
            inputs = data[0]
            mask = None
            if len(data) > 1:
                mask = data[1]

            # dropout matrices for input units
            dp_mask = None
            # dropout matrices for recurrent units
            rec_dp_mask = None

            if 0 < cell.dropout < 1 and cell._dropout_mask is None:
                dp_mask = _generate_dropout_mask(
                    KerasSymbol(mx.sym.ones_like(inputs)),
                    cell.dropout,
                    training=training,
                    count=4)
            if (0 < cell.recurrent_dropout < 1 and
                    cell._recurrent_dropout_mask is None):
                rec_dp_mask = _generate_dropout_mask(
                    KerasSymbol(mx.sym.ones_like(states[0])),
                    cell.recurrent_dropout,
                    training=training,
                    count=4)

            h_tm1 = states[0]  # previous memory state
            c_tm1 = states[1]  # previous carry state

            if cell.implementation == 1:
                if 0 < cell.dropout < 1.:
                    inputs_i = inputs * dp_mask[0].symbol
                    inputs_f = inputs * dp_mask[1].symbol
                    inputs_c = inputs * dp_mask[2].symbol
                    inputs_o = inputs * dp_mask[3].symbol
                else:
                    inputs_i = inputs
                    inputs_f = inputs
                    inputs_c = inputs
                    inputs_o = inputs
                x_i = _dot_rnn(inputs_i, cell.kernel_i.symbol)
                x_f = _dot_rnn(inputs_f, cell.kernel_f.symbol)
                x_c = _dot_rnn(inputs_c, cell.kernel_c.symbol)
                x_o = _dot_rnn(inputs_o, cell.kernel_o.symbol)
                if cell.use_bias:
                    x_i = mx.sym.broadcast_add(x_i, cell.bias_i.symbol)
                    x_f = mx.sym.broadcast_add(x_f, cell.bias_i.symbol)
                    x_c = mx.sym.broadcast_add(x_c, cell.bias_i.symbol)
                    x_o = mx.sym.broadcast_add(x_o, cell.bias_i.symbol)

                if 0 < cell.recurrent_dropout < 1.:
                    h_tm1_i = h_tm1 * rec_dp_mask[0].symbol
                    h_tm1_f = h_tm1 * rec_dp_mask[1].symbol
                    h_tm1_c = h_tm1 * rec_dp_mask[2].symbol
                    h_tm1_o = h_tm1 * rec_dp_mask[3].symbol
                else:
                    h_tm1_i = h_tm1
                    h_tm1_f = h_tm1
                    h_tm1_c = h_tm1
                    h_tm1_o = h_tm1

                i = cell.recurrent_activation(
                    KerasSymbol(x_i + mx.sym.dot(h_tm1_i, cell.recurrent_kernel_i.symbol))).symbol
                f = cell.recurrent_activation(
                    KerasSymbol(x_f + mx.sym.dot(h_tm1_f, cell.recurrent_kernel_f.symbol))).symbol
                c = f * c_tm1 + i * cell.activation(
                    KerasSymbol(x_c + mx.sym.dot(h_tm1_c, cell.recurrent_kernel_c.symbol))).symbol
                o = cell.recurrent_activation(
                    KerasSymbol(x_o + mx.sym.dot(h_tm1_o, cell.recurrent_kernel_o.symbol))).symbol
            else:
                if 0. < cell.dropout < 1.:
                    inputs = inputs * dp_mask[0].symbol
                z = _dot_rnn(inputs, cell.kernel.symbol)
                if 0. < cell.recurrent_dropout < 1.:
                    h_tm1 = h_tm1 * rec_dp_mask[0].symbol
                z = z + _dot_rnn(h_tm1, cell.recurrent_kernel.symbol)
                if cell.use_bias:
                    z = mx.sym.broadcast_add(z, cell.bias.symbol)

                z0 = mx.sym.slice_axis(z, axis=1, begin=0, end=cell.units)
                z1 = mx.sym.slice_axis(z, axis=1, begin=cell.units, end=2 * cell.units)
                z2 = mx.sym.slice_axis(z, axis=1, begin=2 * cell.units, end=3 * cell.units)
                z3 = mx.sym.slice_axis(z, axis=1, begin=3 * cell.units, end=4 * cell.units)

                i = cell.recurrent_activation(KerasSymbol(z0)).symbol
                f = cell.recurrent_activation(KerasSymbol(z1)).symbol
                c = f * c_tm1 + i * cell.activation(KerasSymbol(z2)).symbol
                o = cell.recurrent_activation(KerasSymbol(z3)).symbol
            h = o * cell.activation(KerasSymbol(c)).symbol
            if mask is not None:
                h = mx.sym.where(mask, h, h_tm1)
                c = mx.sym.where(mask, c, c_tm1)
            return h, [h, c]

        def _gru_cell_step(data, states):
            # Refer to GRUCell's call function in keras.layers.recurrent
            h_tm1 = states[0]  # previous memory

            inputs = data[0]
            mask = None
            if len(data) > 1:
                mask = data[1]

            # dropout matrices for input units
            dp_mask = None
            # dropout matrices for recurrent units
            rec_dp_mask = None

            if 0 < cell.dropout < 1 and cell._dropout_mask is None:
                dp_mask = _generate_dropout_mask(
                    KerasSymbol(mx.sym.ones_like(inputs)),
                    cell.dropout,
                    training=training,
                    count=3)
            if (0 < cell.recurrent_dropout < 1 and
                    cell._recurrent_dropout_mask is None):
                rec_dp_mask = _generate_dropout_mask(
                    KerasSymbol(mx.sym.ones_like(h_tm1)),
                    cell.recurrent_dropout,
                    training=training,
                    count=3)

            if cell.implementation == 1:
                if 0. < cell.dropout < 1.:
                    inputs_z = inputs * dp_mask[0].symbol
                    inputs_r = inputs * dp_mask[1].symbol
                    inputs_h = inputs * dp_mask[2].symbol
                else:
                    inputs_z = inputs
                    inputs_r = inputs
                    inputs_h = inputs

                x_z = _dot_rnn(inputs_z, cell.kernel_z.symbol)
                x_r = _dot_rnn(inputs_r, cell.kernel_r.symbol)
                x_h = _dot_rnn(inputs_h, cell.kernel_h.symbol)
                if cell.use_bias:
                    x_z = mx.sym.broadcast_add(x_z, cell.input_bias_z.symbol)
                    x_r = mx.sym.broadcast_add(x_r, cell.input_bias_r.symbol)
                    x_h = mx.sym.broadcast_add(x_h, cell.input_bias_h.symbol)

                if 0. < cell.recurrent_dropout < 1.:
                    h_tm1_z = h_tm1 * rec_dp_mask[0].symbol
                    h_tm1_r = h_tm1 * rec_dp_mask[1].symbol
                    h_tm1_h = h_tm1 * rec_dp_mask[2].symbol
                else:
                    h_tm1_z = h_tm1
                    h_tm1_r = h_tm1
                    h_tm1_h = h_tm1

                recurrent_z = _dot_rnn(h_tm1_z, cell.recurrent_kernel_z.symbol)
                recurrent_r = _dot_rnn(h_tm1_r, cell.recurrent_kernel_r.symbol)
                if cell.reset_after and cell.use_bias:
                    recurrent_z = mx.sym.broadcast_add(recurrent_z, cell.recurrent_bias_z.symbol)
                    recurrent_r = mx.sym.broadcast_add(recurrent_r, cell.recurrent_bias_r.symbol)

                z = cell.recurrent_activation(KerasSymbol(x_z + recurrent_z)).symbol
                r = cell.recurrent_activation(KerasSymbol(x_r + recurrent_r)).symbol

                # reset gate applied after/before matrix multiplication
                if cell.reset_after:
                    recurrent_h = _dot_rnn(h_tm1_h, cell.recurrent_kernel_h.symbol)
                    if cell.use_bias:
                        recurrent_h = mx.sym.broadcast_add(recurrent_h, cell.recurrent_bias_h.symbol)
                    recurrent_h = r * recurrent_h
                else:
                    recurrent_h = _dot_rnn(r * h_tm1_h, cell.recurrent_kernel_h.symbol)

                hh = cell.activation(KerasSymbol(x_h + recurrent_h)).symbol
            else:
                if 0. < cell.dropout < 1.:
                    inputs = inputs * dp_mask[0].symbol

                # inputs projected by all gate matrices at once
                matrix_x = _dot_rnn(inputs, cell.kernel.symbol)
                if cell.use_bias:
                    # biases: bias_z_i, bias_r_i, bias_h_i
                    matrix_x = mx.sym.broadcast_add(matrix_x, cell.input_bias.symbol)

                x_z = mx.sym.slice_axis(matrix_x, axis=1, begin=0, end=cell.units)
                x_r = mx.sym.slice_axis(matrix_x, axis=1, begin=cell.units, end=2 * cell.units)
                x_h = mx.sym.slice_axis(matrix_x, axis=1, begin=2 * cell.units, end=None)

                if 0. < cell.recurrent_dropout < 1.:
                    h_tm1 = h_tm1 * rec_dp_mask[0].symbol

                if cell.reset_after:
                    # hidden state projected by all gate matrices at once
                    matrix_inner = _dot_rnn(h_tm1, cell.recurrent_kernel.symbol)
                    if cell.use_bias:
                        matrix_inner = mx.sym.broadcast_add(matrix_inner, cell.recurrent_bias.symbol)
                else:
                    # hidden state projected separately for update/reset and new
                    matrix_inner = _dot_rnn(h_tm1,
                                            cell.recurrent_kernel[:, :2 * cell.units].symbol)

                recurrent_z = mx.sym.slice_axis(matrix_inner, axis=1, begin=0, end=cell.units)
                recurrent_r = mx.sym.slice_axis(matrix_inner, axis=1, begin=cell.units, end=2 * cell.units)

                z = cell.recurrent_activation(KerasSymbol(x_z + recurrent_z)).symbol
                r = cell.recurrent_activation(KerasSymbol(x_r + recurrent_r)).symbol

                if cell.reset_after:
                    recurrent_h = r * mx.sym.slice_axis(matrix_inner, axis=1, begin=0, end=2 * cell.units)
                else:
                    recurrent_h = _dot_rnn(r * h_tm1,
                                           cell.recurrent_kernel[:, 2 * cell.units:].symbol)

                hh = cell.activation(KerasSymbol(x_h + recurrent_h)).symbol

            # previous and candidate state mixed by update gate
            h = z * h_tm1 + (1 - z) * hh

            if mask is not None:
                h = mx.sym.where(mask, h, h_tm1)
            return h, [h]

        # Reverse the input sequence
        if go_backwards:
            inputs = reverse(inputs, 0)
            if mask is not None:
                mask = reverse(mask, 0)

        # Transpose to time-major, i.e.
        # from (batch, time, ...) to (time, batch, ...)
        ndim = len(inputs.shape)
        axes = [1, 0] + list(range(2, ndim))
        inputs = mx.sym.transpose(inputs.symbol, axes=axes)

        if mask is not None:
            if len(mask.shape) == len(dshape) - 1:
                mask = expand_dims(mask)
            mask = mx.sym.transpose(mask.symbol, axes=axes)
            data = [inputs, mask]
        else:
            data = [inputs]
        states = [state.symbol for state in initial_states] + [constant.symbol for constant in constants]

        # using control flow operators(foreach) in MXNet if not unrolling
        # foreach operator only take step functions with MXNet symbol as input
        # translating step function from Keras RNN cells to pure MXNet
        # define _step according to different type of RNN Cells
        if 'SimpleRNNCell' in type(cell).__name__:
            _step = _simple_rnn_cell_step

        elif 'LSTMCell' in type(cell).__name__:
            _step = _lstm_cell_step

        elif 'GRUCell' in type(cell).__name__:
            _step = _gru_cell_step

        else:
            try:
                # try to support some custom RNN cells
                def _step(data, states):
                    outputs, new_states = step_function(KerasSymbol(data[0]), [KerasSymbol(state) for state in states])
                    if getattr(outputs, '_uses_learning_phase', False):
                        global uses_learning_phase
                        uses_learning_phase = True
                    return outputs.symbol, [new_state.symbol for new_state in new_states]
            except:
                raise NotImplementedError('MXNet Backend: Not supported RNN Cell')
        outputs, states = mx.symbol.contrib.foreach(_step, data, states)
        last_output = KerasSymbol(states[0])
        states = [KerasSymbol(state) for state in states]
        outputs = mx.sym.transpose(outputs, axes)

    outputs = KerasSymbol(outputs)
    last_output._uses_learning_phase = uses_learning_phase
    return last_output, outputs, states


def _dot_rnn(x, y):
    """
    helper function for RNN,
    using pure mxnet symbol for perform K.dot function

    :param x: mxnet symbol, lhs of dot
    :param y: mxnet symbol, rhs of dot
    :return: dot of lhs and rhs
    """
    _, shape, _ = y.infer_shape_partial()
    ndim_y = len(shape[0]) if shape is not None else None

    if ndim_y > 2:
        axis = list(range(ndim_y))
        axis = [axis.pop(-2)] + axis
        y = mx.sym.transpose(y, axes=axis)
    return mx.sym.dot(x, y)


def _generate_dropout_mask(ones, rate, training=None, count=1):
    """
    Generate dropout mask for RNN Cell step functions
    Refer to _generate_dropout_mask in keras/layers/recurrent.py
    """
    def dropped_inputs():
        return dropout(ones, rate)

    if count > 1:
        return [in_train_phase(
            dropped_inputs,
            ones,
            training=training) for _ in range(count)]
    return in_train_phase(
        dropped_inputs,
        ones,
        training=training)


@keras_mxnet_symbol
def switch(condition, then_expression, else_expression):
    """Switches between two operations depending on a scalar value.

    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.

    # Arguments
        condition: scalar tensor (`int` or `bool`).
        then_expression: either a tensor, or a callable that returns a tensor.
        else_expression: either a tensor, or a callable that returns a tensor.

    # Returns
        The selected tensor.

    # Raises
        ValueError: If rank of `condition` is greater than rank of expressions.
    """
    if callable(then_expression):
        then_expression = then_expression()
    if callable(else_expression):
        else_expression = else_expression()
    assert (isinstance(condition, KerasSymbol) and isinstance(then_expression, KerasSymbol) and isinstance(
        else_expression, KerasSymbol))
    return KerasSymbol(
        mx.sym.where(condition.symbol, then_expression.symbol, else_expression.symbol))


def in_train_phase(x, alt, training=None):
    """Selects `x` in train phase, and `alt` otherwise.

    Note that `alt` should have the *same shape* as `x`.

    # Arguments
        x: What to return in train phase
            (tensor or callable that returns a tensor).
        alt: What to return otherwise
            (tensor or callable that returns a tensor).
        training: Optional scalar tensor
            (or Python boolean, or Python integer)
            specifying the learning phase.

    # Returns
        Either `x` or `alt` based on the `training` flag.
        the `training` flag defaults to `K.learning_phase()`.
    """
    uses_learning_phase = False

    if training is None:
        training = learning_phase()
        uses_learning_phase = True

    if training:
        if isinstance(training, KerasSymbol):
            # assume learning phase is a placeholder tensor.
            res = switch(training, x, alt)
        else:
            if callable(x):
                res = x()
            else:
                res = x
            if isinstance(x, KerasSymbol):
                uses_learning_phase = True
    else:
        if callable(alt):
            res = alt()
        else:
            res = alt

    if uses_learning_phase:
        res._uses_learning_phase = True
    return res


def in_test_phase(x, alt, training=None):
    """Selects `x` in test phase, and `alt` otherwise.

    Note that `alt` should have the *same shape* as `x`.

    # Arguments
        x: What to return in test phase
            (tensor or callable that returns a tensor).
        alt: What to return otherwise
            (tensor or callable that returns a tensor).
        training: Optional scalar tensor
            (or Python boolean, or Python integer)
            specifying the learning phase.

    # Returns
        Either `x` or `alt` based on `K.learning_phase`.
    """
    return in_train_phase(alt, x, training=training)


# NN OPERATIONS
@keras_mxnet_symbol
def relu(x, alpha=0., max_value=None, threshold=0.):
    """Rectified linear unit.

    With default values, it returns element-wise `max(x, 0)`.

    # Arguments
        x: A tensor or variable.
        alpha: A scalar, slope of negative section (default=`0.`).
        max_value: Saturation threshold.

    # Returns
        A tensor.
    """
    if alpha != 0.:
        if max_value is None and threshold == 0.:
            return KerasSymbol(mx.sym.LeakyReLU(data=x.symbol, act_type='leaky', slope=alpha))

        if threshold != 0.:
            data = -x + threshold
            negative_part = mx.sym.LeakyReLU(data=data.symbol, act_type='leaky', slope=0.)
        else:
            data = -x
            negative_part = mx.sym.LeakyReLU(data=data.symbol, act_type='leaky', slope=0.)

    clip_max = max_value is not None

    if threshold != 0:
        # computes x for x > threshold else 0
        x = x * cast(greater(x, threshold), floatx())
    else:
        x = KerasSymbol(mx.sym.LeakyReLU(data=x.symbol, act_type='leaky', slope=alpha))

    if clip_max:
        x = KerasSymbol(mx.sym.clip(x.symbol, 0, max_value))

    if alpha != 0:
        negative_part = alpha * negative_part
        x = x - KerasSymbol(negative_part)
    return x


@keras_mxnet_symbol
def elu(x, alpha=1.):
    """Exponential linear unit.

    # Arguments
        x: A tenor or variable to compute the activation function for.
        alpha: A scalar, slope of positive section.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.LeakyReLU(data=x.symbol, act_type='elu', slope=alpha))


@keras_mxnet_symbol
def softmax(x, axis=-1):
    """Softmax of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.softmax(data=x.symbol, axis=axis))


@keras_mxnet_symbol
def softplus(x):
    """Softplus of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.Activation(data=x.symbol, act_type='softrelu'))


@keras_mxnet_symbol
def softsign(x):
    """Softsign of a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(x.symbol / (1 + mx.sym.abs(x.symbol)))


@keras_mxnet_symbol
def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.

    # Returns
        Output tensor.
    """
    output_dimensions = list(range(ndim(output)))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(int_shape(output)))))

    mx_output = output.symbol
    # scale predictions so that the class probabilities of each sample sum to 1
    if from_logits:
        mx_output = mx.sym.softmax(mx_output, axis=axis)
    else:
        mx_output = mx.sym.broadcast_div(mx_output, mx.sym.sum(mx_output,
                                                               axis=axis,
                                                               keepdims=True))

    # clip to prevent NaN's and Inf's
    mx_output = mx.sym.clip(mx_output, a_min=epsilon(), a_max=1.0 - epsilon())
    # calc
    mx_output = - mx.sym.sum(target.symbol * mx.sym.log(mx_output), axis=axis)
    return KerasSymbol(mx_output)


@keras_mxnet_symbol
def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy with integer targets.

    # Arguments
        target: An integer tensor.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.

    # Returns
        Output tensor.
    """
    output_dimensions = list(range(ndim(output)))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(int_shape(output)))))

    mx_output = output.symbol
    # scale predictions so that the class probabilities of each sample sum to 1
    if from_logits:
        mx_output = mx.sym.softmax(mx_output, axis=axis)
    else:
        mx_output = mx.sym.broadcast_div(mx_output, mx.sym.sum(mx_output,
                                                               axis=axis,
                                                               keepdims=True))
    # clip to prevent NaN's and Inf's
    mx_output = mx.sym.clip(mx_output, a_min=epsilon(), a_max=1.0 - epsilon())
    # For this operation, the probability of a given label is considered exclusive.
    mx_output = mx.sym.pick(mx_output, target.symbol, axis=axis, keepdims=True)
    mx_output = - mx.sym.log(mx_output, axis=axis)
    # reshape to input's shape
    return reshape(KerasSymbol(mx_output), target.shape)


@keras_mxnet_symbol
def multi_hot_sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Calculate Categorical crossentropy with a list of integer targets (multi-labels)

    # Arguments
        target: An integer tensor.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.

    # Returns
        Output tensor.

    # Example:
    ```
    # refer to examples/multi_hot_sparse_categorical_crossentropy.py
    # for a multi-label classification problem with 3 classes
    # target with multi labels in normal categorical crossentropy
    >>>target_dense = np.array([[0, 1, 1], [1, 0, 1], [1, 0, 0]])
    # output from softmax remains the same format
    >>>output = np.array([[0.1, 0.4, 0.5],
    >>>                   [0.4, 0.2, 0.4],
    >>>                   [0.7, 0.2, 0.1]])
    # target with multi labels in multi_hot categorical crossentropy
    >>>y_true_np2 = np.array([[1, 2], [0, 2],[0]])
    ```
    """
    if mx.__version__ < '1.3.1':
        raise NotImplementedError('MXNet Backend: multi_hot_sparse_categorical_crossentropy only'
                                  'works with MXNet 1.3.1 or newer, please upgrade MXNet using:'
                                  'pip install --upgrade mxnet')
    output_dimensions = list(range(ndim(output)))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(int_shape(output)))))

    mx_output = output.symbol
    # scale predictions so that the class probabilities of each sample sum to 1
    if from_logits:
        mx_output = mx.sym.softmax(mx_output, axis=axis)
    else:
        mx_output = mx.sym.broadcast_div(mx_output, mx.sym.sum(mx_output,
                                                               axis=axis,
                                                               keepdims=True))
    # clip to prevent NaN's and Inf's
    mx_output = mx.sym.clip(mx_output, a_min=epsilon(), a_max=1.0 - epsilon())

    # using control flow ops to iterate output and take target (true label)
    _step = lambda data, _: (mx.sym.take(data[0], data[1]), [])
    data = [mx_output, target.symbol]
    outputs, _ = mx.symbol.contrib.foreach(_step, data, [])

    # calculate loss
    # check if target is larger than 0, remove padded labels (-1)
    outputs = - mx.sym.sum(mx.sym.broadcast_greater_equal(target.symbol, mx.sym.zeros((1, 1))) *
                           mx.sym.log(outputs), axis=axis)
    return KerasSymbol(outputs)


@keras_mxnet_symbol
def binary_crossentropy(target, output, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.

    # Returns
        A tensor.
    """
    mx_output = output.symbol
    if from_logits:
        mx_output = mx.sym.Activation(mx_output, act_type='sigmoid')
    mx_output = mx.sym.clip(mx_output, a_min=epsilon(), a_max=1 - epsilon())
    mx_output = - (target.symbol * mx.sym.log(mx_output) + (1 - target.symbol) * mx.sym.log(1 - mx_output))
    return KerasSymbol(mx_output)


@keras_mxnet_symbol
def sigmoid(x):
    """Element-wise sigmoid.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.Activation(data=x.symbol, act_type='sigmoid'))


@keras_mxnet_symbol
def hard_sigmoid(x):
    """Segment-wise linear approximation of sigmoid.

    Faster than sigmoid.
    Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
    In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.clip(data=(0.2 * x.symbol + 0.5), a_min=0., a_max=1.))


@keras_mxnet_symbol
def tanh(x):
    """Element-wise tanh.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return KerasSymbol(mx.sym.tanh(data=x.symbol))


@keras_mxnet_symbol
def dropout(x, level, noise_shape=None, seed=None):
    """Sets entries in `x` to zero at random, while scaling the entire tensor.

    # Arguments
        x: tensor
        level: fraction of the entries in the tensor
            that will be set to 0.
        noise_shape: shape for randomly generated keep/drop flags,
            must be broadcastable to the shape of `x`
        seed: random seed to ensure determinism.

    # Returns
        A tensor.
    """
    if not 0 <= level <= 1:
        raise ValueError('MXNet Backend: Invalid level provided for dropout `{0}`. '
                         'Expected between 0 and 1.'.format(level))
    if seed:
        mx.random.seed(seed)
    else:
        mx.random.seed(int(10e6))
    name = _prepare_name(None, 'dropout')
    return KerasSymbol(mx.sym.Dropout(data=x.symbol, p=level, name=name))


@keras_mxnet_symbol
def l2_normalize(x, axis=None):
    """Normalizes a tensor wrt the L2 norm alongside the specified axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform normalization.

    # Returns
        A tensor.
    """
    norm = mx.sym.sqrt(data=mx.sym.sum(data=mx.sym.square(data=x.symbol),
                                       axis=axis, keepdims=True))
    return KerasSymbol(mx.sym.broadcast_div(x.symbol, norm))


@keras_mxnet_symbol
def in_top_k(predictions, targets, k):
    """Returns whether the `targets` are in the top `k` `predictions`.

    # Arguments
        predictions: A tensor of shape `(batch_size, classes)` and type `float32`.
        targets: A 1D tensor of length `batch_size` and type `int32` or `int64`.
        k: An `int`, number of top elements to consider.

    # Returns
        A 1D tensor of length `batch_size` and type `bool`.
        `output[i]` is `True` if `predictions[i, targets[i]]` is within top-`k`
        values of `predictions[i]`.
    """
    # MXNet do not return boolean. It returns 0s and 1s.
    targets_sym = mx.sym.Cast(targets.symbol, dtype='int32')
    topk_sym = mx.sym.Cast(mx.sym.topk(data=predictions.symbol, k=k, ret_typ='mask'),
                           dtype='uint8')
    return KerasSymbol(mx.sym.pick(topk_sym, targets_sym))


# CONVOLUTIONS
def conv1d(x, kernel, strides=1, padding='valid',
           data_format=None, dilation_rate=1):
    """1D convolution.

    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: stride integer.
        padding: string, `"same"`, `"causal"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        dilation_rate: integer dilate rate.

    # Returns
        A tensor, result of 1D convolution.
    """
    if data_format is None:
        data_format = image_data_format()
    _validate_data_format(data_format)

    if padding not in {'same', 'valid', 'causal'}:
        raise ValueError('MXNet Backend: `padding` should be either `same`, `valid` or `causal`.')

    if ndim(x) != 3:
        raise ValueError('MXNet Backend: Conv1D with causal padding is supported only for 3D tensors')

    # Causal requires temporal padding.
    # Add temporal padding
    kernel_shape = kernel.shape
    if padding == 'causal':
        pad = dilation_rate * (kernel_shape[0] - 1)
        x = temporal_padding(x, (pad, 0))
        padding = 'valid'

    if hasattr(x, '_keras_shape'):
        shape = x._keras_shape
    else:
        shape = None

    if data_format == 'channels_last':
        # X original shape (batch, length, input_dim)
        # Add a dimension to X to Make it (batch, length, 1, input_dim)
        x = expand_dims(x, axis=2)
        # Add dimension to kernel
        # for channels last: kernel_shape = kernel_size + (input_dim, filters)
        # it will become: (kernel_size, 1, input_dim, filters)
        kernel = expand_dims(kernel, axis=1)
        # update x._keras_shape
        if shape is not None:
            x._keras_shape = (shape[0], shape[1], 1, shape[2])
    elif data_format == 'channels_first':
        # X original shape (batch, input_dim, length)
        # Add a dimension to X to make it (batch, input_dim, length, 1)
        x = expand_dims(x, axis=3)
        # Add dimension to kernel
        # for channels first: kernel_shape = (filters, input_dim) + kernel_size
        # it will become: (filters, input_dim, kernel_size, 1)
        kernel = expand_dims(kernel, axis=3)
        if shape is not None:
            x._keras_shape = (shape[0], shape[1], shape[2], 1)

    # update dilation rate, strides
    dilation_rate = (dilation_rate, 1)
    strides = (strides, 1)

    output = _convnd(x, kernel, name='conv1d', strides=strides, filter_dilation=dilation_rate,
                     padding_mode=padding, data_format=data_format)

    # Remove added extra dimension
    # remove added dim
    if data_format == 'channels_last':
        output = squeeze(output, axis=2)
    else:
        output = squeeze(output, axis=3)
    return output


def conv2d(x, kernel, strides=(1, 1), padding='valid',
           data_format=None, dilation_rate=(1, 1)):
    """2D convolution.

    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 2 integers.

    # Returns
        A tensor, result of 2D convolution.

    # Raises
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.
    """
    if data_format is None:
        data_format = image_data_format()
    _validate_data_format(data_format)

    if padding not in {'same', 'valid'}:
        raise ValueError('MXNet Backend: `padding` should be either `same` or `valid`.')

    return _convnd(x, kernel, name='conv2d', strides=strides, filter_dilation=dilation_rate,
                   padding_mode=padding, data_format=data_format)


def conv2d_transpose(x, kernel, output_shape, strides=(1, 1),
                     padding='valid', data_format=None, dilation_rate=(1, 1)):
    """2D deconvolution (i.e. transposed convolution).

    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.

    # Returns
        A tensor, result of transposed 2D convolution.

    # Raises
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.

    # Example (detailed example refer to mnist_denoising_autoencoder.py):
    ```
        >>>from keras.models import Sequential
        >>>from keras.layers import Conv2DTranspose
        >>>model = Sequential()
        >>>model.add(Conv2DTranspose(32, (3, 3), activation='relu',
        >>>          input_shape=(100, 100, 3)))
        >>>model.summary()
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        conv2d_transpose_2 (Conv2DTr (None, 32, 102, 5)        28832
        =================================================================
    ```
    """
    if data_format is None:
        data_format = image_data_format()
    _validate_data_format(data_format)

    if padding not in {'same', 'valid'}:
        raise ValueError('MXNet Backend: `padding` should be either `same` or `valid`.')

    return _convnd_transpose(x, kernel, output_shape, name='conv2d_transpose',
                             strides=strides, data_format=data_format, dilation_rate=dilation_rate)


def separable_conv1d(x, depthwise_kernel, pointwise_kernel, strides=1,
                     padding='valid', data_format=None, dilation_rate=1):
    """1D convolution with separable filters.

    # Arguments
        x: input tensor
        depthwise_kernel: convolution kernel for the depthwise convolution.
        pointwise_kernel: kernel for the 1x1 convolution.
        strides: strides integer.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        dilation_rate: integer dilation rate.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    raise NotImplementedError('MXNet Backend: Separable Conv1D not supported yet!')


def separable_conv2d(x, depthwise_kernel, pointwise_kernel, strides=(1, 1),
                     padding='valid', data_format=None, dilation_rate=(1, 1)):
    """2D convolution with separable filters.

    # Arguments
        x: input tensor
        depthwise_kernel: convolution kernel for the depthwise convolution.
        pointwise_kernel: kernel for the 1x1 convolution.
        strides: strides tuple (length 2).
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        dilation_rate: tuple of integers,
            dilation rates for the separable convolution.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.
    """
    """
    # mathematical implementation of complete separable conv2d
    return _sp_convnd(x,  depthwise_kernel, pointwise_kernel, strides=strides,
                      padding_mode=padding, data_format=data_format,
                      filter_dilation=dilation_rate)
    """
    # depthwise conv2d
    dw_conv = depthwise_conv2d(x, depthwise_kernel, strides=strides, padding=padding, data_format=data_format,
                               dilation_rate=dilation_rate)
    # pointwise conv2d, strides is always (1, 1)
    return _convnd(dw_conv, kernel=pointwise_kernel, strides=(1, 1), padding_mode=padding, data_format=data_format,
                   filter_dilation=dilation_rate)


def depthwise_conv2d(x, depthwise_kernel, strides=(1, 1), padding='valid',
                     data_format=None, dilation_rate=(1, 1)):
    """2D convolution with separable filters.

    # Arguments
        x: input tensor
        depthwise_kernel: convolution kernel for the depthwise convolution.
        strides: strides tuple (length 2).
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        dilation_rate: tuple of integers,
            dilation rates for the separable convolution.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.
    """
    if data_format is None:
        data_format = image_data_format()
    _validate_data_format(data_format)

    if padding not in {'same', 'valid'}:
        raise ValueError('MXNet Backend: `padding` should be either `same` or `valid`.')

    dw_out = _dw_conv(x, depthwise_kernel, name='dw_conv2d', strides=strides, filter_dilation=dilation_rate,
                      padding_mode=padding, data_format=data_format)
    return dw_out


def conv3d(x, kernel, strides=(1, 1, 1), padding='valid',
           data_format=None, dilation_rate=(1, 1, 1)):
    """3D convolution.

    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 3 integers.

    # Returns
        A tensor, result of 3D convolution.

    # Raises
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.
    """
    if data_format is None:
        data_format = image_data_format()
    _validate_data_format(data_format)

    if padding not in {'same', 'valid'}:
        raise ValueError('MXNet Backend: `padding` should be either `same` or `valid`.')

    return _convnd(x, kernel, name='conv3d', strides=strides, filter_dilation=dilation_rate,
                   padding_mode=padding, data_format=data_format)


def conv3d_transpose(x, kernel, output_shape, strides=(1, 1, 1),
                     padding='valid', data_format=None):
    """3D deconvolution (i.e. transposed convolution).

    # Arguments
        x: input tensor.
        kernel: kernel tensor.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        padding: string, "same" or "valid".
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.

    # Returns
        A tensor, result of transposed 3D convolution.

    # Raises
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.
    """
    # MXNet only support Conv3D with GPU and CUDNN
    gpus = mx.test_utils.list_gpus()
    if gpus and len(gpus) > 0:
        if data_format is None:
            data_format = image_data_format()
        _validate_data_format(data_format)

        if padding not in {'same', 'valid'}:
            raise ValueError('MXNet Backend: `padding` should be either `same` or `valid`.')

        return _convnd_transpose(x, kernel, output_shape, name='conv3d_transpose', strides=strides,
                                 data_format=data_format)
    else:
        raise NotImplementedError('MXNet Backend: Conv3D Transpose is only supported on GPU with CUDNN')


@keras_mxnet_symbol
def pool2d(x, pool_size, strides=(1, 1),
           padding='valid', data_format=None,
           pool_mode='max'):
    """2D Pooling.

    # Arguments
        x: Tensor or variable.
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        pool_mode: string, `"max"` or `"avg"`.

    # Returns
        A tensor, result of 2D pooling.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
        ValueError: if `pool_mode` is neither `"max"` or `"avg"`.
    """

    return _poolnd(x=x, name='pool2d', pool_size=pool_size, strides=strides, padding_mode=padding,
                   data_format=data_format, pool_mode=pool_mode)


@keras_mxnet_symbol
def pool3d(x, pool_size, strides=(1, 1, 1), padding='valid',
           data_format=None, pool_mode='max'):
    """3D Pooling.

    # Arguments
        x: Tensor or variable.
        pool_size: tuple of 3 integers.
        strides: tuple of 3 integers.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        pool_mode: string, `"max"` or `"avg"`.

    # Returns
        A tensor, result of 3D pooling.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
        ValueError: if `pool_mode` is neither `"max"` or `"avg"`.
    """
    return _poolnd(x=x, name='pool3d', pool_size=pool_size, strides=strides, padding_mode=padding,
                   data_format=data_format, pool_mode=pool_mode)


@keras_mxnet_symbol
def bias_add(x, bias, data_format='channels_last'):
    """Adds a bias vector to a tensor.

    # Arguments
        x: Tensor or variable.
        bias: Bias tensor to add.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        Output tensor.

    # Raises
        ValueError: In one of the two cases below:
                    1. invalid `data_format` argument.
                    2. invalid bias shape.
                       the bias should be either a vector or
                       a tensor with ndim(x) - 1 dimension
    """
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('MXNet Backend: Unknown data_format ' + str(data_format))
    bias_shape = int_shape(bias)
    x_dim = ndim(x)
    if len(bias_shape) != 1 and len(bias_shape) != x_dim - 1:
        raise ValueError('MXNet Backend: Unexpected bias dimensions %d, expect to be 1 or %d dimensions'
                         % (len(bias_shape), x_dim))
    if x_dim == 5:
        if data_format == 'channels_first':
            if len(bias_shape) == 1:
                x += reshape(bias, (1, bias_shape[0], 1, 1, 1))
            else:
                x += reshape(bias, (1, bias_shape[3]) + bias_shape[:3])
        elif data_format == 'channels_last':
            if len(bias_shape) == 1:
                x += bias
            else:
                x += reshape(bias, (1,) + bias_shape)
    elif x_dim == 4:
        if data_format == 'channels_first':
            if len(bias_shape) == 1:
                x += reshape(bias, (1, bias_shape[0], 1, 1))
            else:
                x += reshape(bias, (1, bias_shape[2]) + bias_shape[:2])
        elif data_format == 'channels_last':
            if len(bias_shape) == 1:
                x += bias
            else:
                x += reshape(bias, (1,) + bias_shape)
    elif x_dim == 3:
        if data_format == 'channels_first':
            if len(bias_shape) == 1:
                x += reshape(bias, (1, bias_shape[0], 1))
            else:
                x += reshape(bias, (1, bias_shape[1], bias_shape[0]))
        elif data_format == 'channels_last':
            if len(bias_shape) == 1:
                x += bias
            else:
                x += reshape(bias, (1,) + bias_shape)
    else:
        x += bias
    return x


# RANDOMNESS
@keras_mxnet_symbol
def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """Returns a tensor with normal distribution of values.

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        mean: A float, mean of the normal distribution to draw samples.
        stddev: A float, standard deviation of the normal distribution
            to draw samples.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    shape = tuple([0 if dim is None else dim for dim in shape])
    if seed:
        mx.random.seed(seed)
    else:
        mx.random.seed(int(10e6))
    sym = mx.sym.random.normal(shape=shape, loc=mean, scale=stddev, dtype=dtype)
    return KerasSymbol(sym)


@keras_mxnet_symbol
def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    """Returns a tensor with uniform distribution of values.

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        minval: A float, lower boundary of the uniform distribution
            to draw samples.
        maxval: A float, upper boundary of the uniform distribution
            to draw samples.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    shape = tuple([0 if dim is None else dim for dim in shape])
    if seed:
        mx.random.seed(seed)
    else:
        mx.random.seed(int(10e6))
    sym = mx.sym.random.uniform(shape=shape, low=minval, high=maxval, dtype=dtype)
    return KerasSymbol(sym)


@keras_mxnet_symbol
def random_binomial(shape, p=0.0, dtype=None, seed=None):
    """Returns a tensor with random binomial distribution of values.

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        p: A float, `0. <= p <= 1`, probability of binomial distribution.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    shape = tuple([0 if dim is None else dim for dim in shape])
    if seed:
        mx.random.seed(seed)
    else:
        mx.random.seed(int(10e6))
    sym = mx.sym.random.uniform(shape=shape, low=0., high=1., dtype=dtype)
    sym = mx.sym.where(sym <= p,
                       mx.sym.ones(shape=shape, dtype=dtype),
                       mx.sym.zeros(shape=shape, dtype=dtype))
    return KerasSymbol(sym)


@keras_mxnet_symbol
def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """Returns a tensor with truncated random normal distribution of values.

    The generated values follow a normal distribution
    with specified mean and standard deviation,
    except that values whose magnitude is more than
    two standard deviations from the mean are dropped and re-picked.

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        mean: Mean of the values.
        stddev: Standard deviation of the values.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    """
    if dtype is None:
        dtype = floatx()
    dtype = _convert_string_dtype(dtype)
    shape = tuple([0 if dim is None else dim for dim in shape])
    if seed:
        mx.random.seed(seed)
    else:
        mx.random.seed(int(10e6))
    ran = mx.sym.random.normal(shape=shape, loc=mean, scale=stddev, dtype=dtype)
    res = mx.sym.clip(data=ran, a_min=mean - 2 * stddev, a_max=mean + 2 * stddev)
    return KerasSymbol(res)


# CTC(Connectionist Temporal Classification)

def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    """Runs CTC loss algorithm on each batch element.

    # Arguments
        y_true: tensor `(samples, max_string_length)`
            containing the truth labels.
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_pred`.
        label_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_true`.

    # Returns
        Tensor with shape (samples,1) containing the
            CTC loss of each element.
    """
    raise NotImplementedError("MXNet Backend: CTC is not supported yet.")


# HIGH ORDER FUNCTIONS

def map_fn(fn, elems, name=None, dtype=None):
    """Map the function fn over the elements elems and return the outputs.

    # Arguments
        fn: Callable that will be called upon each element in elems
        elems: tensor
        name: A string name for the map node in the graph
        dtype: Output data type.

    # Returns
        Tensor with dtype `dtype`.
    """
    raise NotImplementedError('MXNet Backend: map_fn operator is not supported yet.')


def foldl(fn, elems, initializer=None, name=None):
    """Reduce elems using fn to combine them from left to right.

    # Arguments
        fn: Callable that will be called upon each element in elems and an
            accumulator, for instance `lambda acc, x: acc + x`
        elems: tensor
        initializer: The first value used (`elems[0]` in case of None)
        name: A string name for the foldl node in the graph

    # Returns
        Tensor with same type and shape as `initializer`.
    """
    raise NotImplementedError('MXNet Backend: foldl operator is not supported yet.')


def foldr(fn, elems, initializer=None, name=None):
    """Reduce elems using fn to combine them from right to left.

    # Arguments
        fn: Callable that will be called upon each element in elems and an
            accumulator, for instance `lambda acc, x: acc + x`
        elems: tensor
        initializer: The first value used (`elems[-1]` in case of None)
        name: A string name for the foldr node in the graph

    # Returns
        Tensor with same type and shape as `initializer`.
    """
    raise NotImplementedError('MXNet Backend: foldr operator is not supported yet.')


def local_conv1d(inputs, kernel, kernel_size, strides, data_format=None):
    """Apply 1D conv with un-shared weights.

    # Arguments
        inputs: 3D tensor with shape: (batch_size, steps, input_dim)
        kernel: the unshared weight for convolution,
                with shape (output_length, feature_dim, filters)
        kernel_size: a tuple of a single integer,
                     specifying the length of the 1D convolution window
        strides: a tuple of a single integer,
                 specifying the stride length of the convolution
        data_format: the data format, channels_first or channels_last

    # Returns
        the tensor after 1d conv with un-shared weights, with shape
        (batch_size, output_length, filters)

    # Raises
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.
    """
    raise NotImplementedError('MXNet Backend: local_conv1d operator is not supported yet.')


def local_conv2d(inputs, kernel, kernel_size, strides, output_shape, data_format=None):
    """Apply 2D conv with un-shared weights.

    # Arguments
        inputs: 4D tensor with shape:
                (batch_size, filters, new_rows, new_cols)
                if data_format='channels_first'
                or 4D tensor with shape:
                (batch_size, new_rows, new_cols, filters)
                if data_format='channels_last'.
        kernel: the unshared weight for convolution,
                with shape (output_items, feature_dim, filters)
        kernel_size: a tuple of 2 integers, specifying the
                     width and height of the 2D convolution window.
        strides: a tuple of 2 integers, specifying the strides
                 of the convolution along the width and height.
        output_shape: a tuple with (output_row, output_col)
        data_format: the data format, channels_first or channels_last

    # Returns
        A 4d tensor with shape:
        (batch_size, filters, new_rows, new_cols)
        if data_format='channels_first'
        or 4D tensor with shape:
        (batch_size, new_rows, new_cols, filters)
        if data_format='channels_last'.

    # Raises
        ValueError: if `data_format` is neither
                    `channels_last` or `channels_first`.
    """
    raise NotImplementedError('MXNet Backend: local_conv2d operator is not supported yet.')


# Other Common Utilities
def get_uid(prefix=''):
    """Provides a unique UID given a string prefix.

    # Arguments
        prefix: string.

    # Returns
        An integer.

    # Example
    ```
        >>> keras.backend.get_uid('dense')
        >>> 1
        >>> keras.backend.get_uid('dense')
        >>> 2
    ```
    """
    _UID_PREFIXES[prefix] += 1
    return _UID_PREFIXES[prefix]


def reset_uids():
    global _UID_PREFIXES
    _UID_PREFIXES = defaultdict(int)


# MXNet backend helper functions
def _prepare_name(name, default):
    """prepares name for the variables

    # Arguments
        name: Expected name of the variable.
        default: If name is None, default is used as name of the variable.

    # Returns
        A unique name for the variable.
    """

    prefix = '/'.join(NAME_SCOPE_STACK)
    if name is None:
        name = prefix + '/' + default
    else:
        name = prefix + '/' + name
    name += "%d" % get_uid(name)
    return name


class KerasSymbol(object):
    """Wrapper on top of MXNet symbol objects. Helps to encapsulate symbolic
    computation graph and binding values.
    """

    def __init__(self, mxnet_symbol, stype='default', neighbors=None, is_var=False):
        if not isinstance(mxnet_symbol, mx.sym.Symbol):
            raise TypeError('MXNet Backend: Please use a MXNet Symbol to instantiate '
                            'a Keras Symbol.')
        if is_var:
            self._train_sym = mxnet_symbol
            self._pred_sym = mxnet_symbol
        else:
            self._train_sym = mxnet_symbol if learning_phase() else None
            self._pred_sym = None if learning_phase() else mxnet_symbol
        self._name = None
        self._neighbors = []

        if neighbors:
            for node in neighbors:
                self.add_neighbor(node)
        self._bind_values = {}
        self.tensor = None
        self._stype = stype

    def bind(self, data):
        if not hasattr(self, 'tensor'):
            self.tensor[:] = data
        else:
            self.tensor = data
        if self.name in self._bind_values:
            assert self._bind_values[self.name].shape == data.shape, \
                'Redefinition of variable %s' % self.name
            assert self._bind_values[self.name].dtype == data.dtype, \
                'Redefinition of variable %s' % self.name
            if _MODEL is not None and self.name in _MODEL._args:
                _MODEL._set_weights({self.name: data}, {})
            if _MODEL is not None and self.name in _MODEL._auxs:
                _MODEL._set_weights({}, {self.name: data})
            else:
                self._bind_values[self.name][:] = data
        else:
            self._bind_values[self.name] = data

    def add_neighbor(self, x):
        if not isinstance(x, KerasSymbol):
            return
        if x not in self._neighbors:
            self._neighbors.append(x)
            x.add_neighbor(self)

    def get_neighbor(self):
        return self._neighbors

    def get_bind_values(self):
        return self._bind_values

    def get_stype(self):
        return self._stype

    @property
    def symbol(self):
        sym = self._train_sym if learning_phase() else self._pred_sym
        assert sym is not None, '[Debug Info] %s, %s' % (self._train_sym, self._pred_sym)
        return sym

    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return self.symbol.name

    @property
    def dtype(self):
        return self._get_type()

    @property
    def shape(self):
        return self._get_shape()

    def eval(self):
        return self.tensor

    def _get_shape(self):
        if hasattr(self, '_keras_shape'):
            return self._keras_shape
        else:
            _, out_shape, _ = self.symbol.infer_shape_partial()
            return out_shape[0]

    def _get_type(self):
        _, out_type, _ = self.symbol.infer_type()
        t = out_type[0]
        return _convert_dtype_string(t)

    @keras_mxnet_symbol
    def __getitem__(self, in_slice):
        begin = []
        end = []
        # in_slice should be a tuple or list of slice() constructor
        # Convert it to a tuple iterator for single dimensional bias Tensors
        if not isinstance(in_slice, (list, tuple)):
            in_slice = (in_slice,)

        # MXNet sym.slice() operator does not support slicing the complete axis with '-1' indexing.
        # See here for more details - https://github.com/awslabs/keras-apache-mxnet/issues/113
        # This is a workaround solution, where:
        # 1. We slice all the elements with indexing -1
        # 2. Squeeze the axis to get the effect of slicing the complete axis.
        # Ex: if input.shape (2,2,3) and indexing is like input[:, -1, :]
        #     1. After step 1, result = (2,1,3)
        #     2. After step 2, result = (2,3) which is same as Numpy and other backend behavior.
        slice_axis = False
        sliced_dim = [d for d in range(len(in_slice)) if in_slice[d] == -1]
        for i in in_slice:
            if isinstance(i, int):
                # Want to slice the complete axis
                if i < -1:
                    raise NotImplementedError('MXNet Backend: Does not support slicing with < -1 indexing. Given - ', i)
                elif i == -1:
                    begin.append(-1)
                    end.append(None)
                    slice_axis = True
                else:
                    begin.append(i)
                    end.append(i + 1)
            elif isinstance(i, slice):
                assert i.step is None or i.step == 1
                begin.append(i.start)
                end.append(i.stop)
            else:
                raise AttributeError('MXNet Backend: KerasSymbol __getitem__ error.')
        sliced_res = mx.sym.slice(self.symbol, begin=tuple(begin),
                                  end=tuple(end))
        if slice_axis:
            num_sliced_axis = 0
            for dim in sliced_dim:
                sliced_res = mx.sym.squeeze(sliced_res, axis=dim - num_sliced_axis)
                num_sliced_axis += 1

        sliced_keras_symbol = KerasSymbol(sliced_res, neighbors=[self])

        # MXNet does not support Scalars. To overcome that,
        # we have introduced a logic to identify (1, ) shaped tensor as vector or scalar.
        # See eval() function for more details.
        # Output of slicing is always a Vector. Without this flag, eval(a tensor with (1,) shape)
        # will be returned as Scalar.
        sliced_keras_symbol._is_vector = True
        return sliced_keras_symbol

    @keras_mxnet_symbol
    def __abs__(self):
        return KerasSymbol(mx.sym.abs(self.symbol), neighbors=[self])

    @keras_mxnet_symbol
    def __add__(self, other):
        if isinstance(other, KerasSymbol):
            return KerasSymbol(
                mx.sym.broadcast_add(
                    lhs=self.symbol,
                    rhs=other.symbol))
        else:
            return KerasSymbol(
                self.symbol + other)

    @keras_mxnet_symbol
    def __radd__(self, other):
        return self.__add__(other)

    @keras_mxnet_symbol
    def __sub__(self, other):
        if isinstance(other, KerasSymbol):
            return KerasSymbol(
                mx.sym.broadcast_sub(
                    lhs=self.symbol,
                    rhs=other.symbol))
        else:
            return KerasSymbol(self.symbol - other)

    @keras_mxnet_symbol
    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    @keras_mxnet_symbol
    def __neg__(self):
        return KerasSymbol(self.symbol * (-1.0), neighbors=[self])

    @keras_mxnet_symbol
    def __div__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(
                self.symbol / other)
        else:
            return KerasSymbol(
                mx.sym.broadcast_div(
                    lhs=self.symbol,
                    rhs=other.symbol))

    @keras_mxnet_symbol
    def __truediv__(self, other):
        return self.__div__(other)

    @keras_mxnet_symbol
    def __itruediv__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(
                self.symbol / other)
        else:
            return KerasSymbol(
                mx.sym.broadcast_div(
                    lhs=self.symbol,
                    rhs=other.symbol))

    @keras_mxnet_symbol
    def __mul__(self, other):
        if isinstance(other, KerasSymbol):
            return KerasSymbol(
                mx.sym.broadcast_mul(
                    lhs=self.symbol,
                    rhs=other.symbol))
        else:
            return KerasSymbol(self.symbol * other)

    @keras_mxnet_symbol
    def __rmul__(self, other):
        return self.__mul__(other)

    @keras_mxnet_symbol
    def __gt__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(self.symbol > other)
        else:
            return KerasSymbol(
                mx.sym.broadcast_greater(
                    lhs=self.symbol,
                    rhs=other.symbol))

    @keras_mxnet_symbol
    def __ge__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(self.symbol >= other)
        else:
            return KerasSymbol(
                mx.sym.broadcast_greater_equal(
                    lhs=self.symbol,
                    rhs=other.symbol))

    @keras_mxnet_symbol
    def __lt__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(
                self.symbol < other)
        else:
            return KerasSymbol(
                mx.sym.broadcast_lesser(
                    lhs=self.symbol,
                    rhs=other.symbol))

    @keras_mxnet_symbol
    def __le__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(
                self.symbol <= other)
        else:
            return KerasSymbol(
                mx.sym.broadcast_lesser_equal(
                    lhs=self.symbol,
                    rhs=other.symbol))

    @keras_mxnet_symbol
    def __gt__(self, other):
        if isinstance(other, Number):
            return KerasSymbol(
                self.symbol > other)
        else:
            return KerasSymbol(
                mx.sym.broadcast_greater(
                    lhs=self.symbol,
                    rhs=other.symbol))

    @keras_mxnet_symbol
    def __pow__(self, power, modulo=None):
        return KerasSymbol(self.symbol.__pow__(power), neighbors=[self])

    def __repr__(self):
        return self.symbol.name + ':[tensor=' + str(hasattr(self, 'tensor')) + ' dtype=' + self.dtype + ']'

    def __str__(self):
        return 'Symbol: %s' % self.symbol.name


def dfs_get_bind_values(node_start):
    """Performs Depth First Search (DFS) on the symbolic computation graph and
    returns the binding Tensor values.

     # Arguments
        node_start: MXNet Symbol. Starting node of the computation graph.

     # Returns
        List of binding Tensor values in the computation graph.
    """
    stack_list = []
    visited = set()
    stack_list.append(node_start)
    while len(stack_list) > 0:
        cur_node = stack_list.pop()
        if cur_node in visited:
            continue
        visited.add(cur_node)
        next_nodes = cur_node.get_neighbor()
        for i in next_nodes:
            if i in visited:
                continue
            else:
                stack_list.append(i)
    bind_values = {}
    for key in visited:
        bind_values.update(key.get_bind_values())
    return bind_values


def _forward_pass(x):
    bind_values = dfs_get_bind_values(x)
    executor = x.symbol.simple_bind(mx.cpu(), grad_req='null')
    for v in executor.arg_dict:
        if v in bind_values.keys():
            bind_values[v].copyto(executor.arg_dict[v])
    outputs = executor.forward(is_train=learning_phase())
    return outputs


def _keras_variable(name, shape, dtype, stype='default', is_vector=False, **kwargs):
    if dtype is None:
        dtype = floatx()
    v = mx.sym.Variable(name, shape=shape, stype=stype, dtype=dtype, **kwargs)
    ret = KerasSymbol(v, stype=stype, is_var=True)

    # MXNet does not support Scalars. Shape of a Scalar Tensor with MXNet is (1, ) instead of ().
    # This flag is used to identify Scalar Keras Variable versus a Tensor of shape (1, ) i.e., vector.
    # For example - bias vector shape is (1, ) when number of neuron in a dense layer is 1.
    # This is useful in K.eval() function to return as is (1, ) or return variable[0] to match expectation of ().
    if is_vector:
        ret._is_vector = is_vector
    return ret


def _convert_string_dtype(dtype):
    """Get the type from a string.

    # Arguments
        dtype: A string representation of a type.

    # Returns
        The type requested.

    # Raises
        ValueError: if `dtype` is not supported.
    """
    if isinstance(dtype, np.dtype):
        # If user has passed the np.dtype, fetch and return the np type.
        ret_type = dtype.type
    elif isinstance(dtype, type):
        # If user has passed the np type, just return it.
        ret_type = dtype
    else:
        # If string name of the type, convert it to a type.
        mapping = {'float16': np.float16,
                   'float32': np.float32,
                   'float64': np.float64,
                   'int16': np.int16,
                   'int8': np.int8,
                   'int32': np.int32,
                   'int64': np.int64,
                   'uint8': np.int8,
                   'uint16': np.uint16}

        if dtype not in mapping:
            raise ValueError('MXNet Backend: Unsupported dtype:', dtype)
        ret_type = mapping[dtype]
    return ret_type


def _convert_dtype_string(dtype):
    """Get the String from type.

    # Arguments
        dtype: Type.

    # Returns
       A stromg representation of a type.

    # Raises
        ValueError: if `dtype` is not supported.
    """
    mapping = {np.float16: 'float16',
               np.float32: 'float32',
               np.float64: 'float64',
               np.int16: 'int16',
               np.int32: 'int32',
               np.int64: 'int64',
               np.int8: 'uint8',
               np.uint8: 'uint8',
               np.uint16: 'uint16'}

    if dtype not in mapping:
        raise ValueError('MXNet Backend: Unsupported dtype:', dtype)
    return mapping[dtype]


def _normalize_axis(axis, ndim):
    if isinstance(axis, tuple):
        axis = list(axis)
    if ndim is None or ndim == 0:
        return axis

    if isinstance(axis, list):
        for i, a in enumerate(axis):
            if a is not None and a < 0:
                axis[i] = a % ndim
    elif axis is None:
        return ()
    else:
        if axis < 0:
            axis = axis % ndim
    return axis


def _validate_data_format(data_format):
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('MXNet Backend: Unknown data_format ' + str(data_format))


def _validate_pool_mode(pool_mode):
    if pool_mode not in {'max', 'avg'}:
        raise ValueError('MXNet Backend: `pool_mode` should be either `max` or `avg`. '
                         'Given - ' + str(pool_mode))


def _validate_padding_mode(padding):
    if padding not in {'same', 'valid', 'full'}:
        raise ValueError('MXNet Backend: `padding` should be either `same`, `full`, `valid`. '
                         'Given - ' + str(padding))


# Convolution Helpers

# Preprocess and Postprocess helper functions to manage data_format
# TF uses the channels_last and MXNet needs channels_first,
# preprocess: (rows, cols, input_depth, depth) => (depth, input_depth, rows, cols)
# postprocess: (depth, input_depth, rows, cols) => (rows, cols, input_depth, depth)


def _calculate_conv_output_size(input_length, filter_size, padding, stride,
                                dilation=1):
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'causal':
        output_length = input_length
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride


@keras_mxnet_symbol
def _preprocess_convnd_input(data_var, data_format):
    if data_format == 'channels_last' and ndim(data_var) > 3:
        axes = list(range(ndim(data_var)))
        axes.insert(1, axes.pop(-1))  # make it channels_first format
        data_var = KerasSymbol(mx.sym.transpose(data=data_var.symbol, axes=axes))
    return data_var


@keras_mxnet_symbol
def _postprocess_convnd_output(x, data_format):
    if data_format == 'channels_last' and ndim(x) > 3:
        idx = list(range(ndim(x)))
        # Convert result back to channels_last format
        idx.append(idx.pop(1))
        x = KerasSymbol(mx.sym.transpose(data=x.symbol, axes=idx))
    return x


@keras_mxnet_symbol
def _preprocess_convnd_kernel(kernel, data_format):
    # If data_format is channels_last, Kernel is TF kernel shape:
    #   2-D: (rows, cols, input_depth, depth)
    #   3-D: (kernel_depth, kernel_rows, kernel_cols, input_depth, depth)
    # Convert it to MXNet kernel shape:
    #   2-D: (depth, input_depth, rows, cols)
    #   3-D: (depth, input_depth, kernel_depth, kernel_rows, kernel_cols)
    #
    if data_format == 'channels_last':
        if len(kernel.shape) > 4:
            kernel = KerasSymbol(mx.sym.transpose(data=kernel.symbol, axes=(4, 3, 0, 1, 2)))
        elif len(kernel.shape) > 3:
            kernel = KerasSymbol(mx.sym.transpose(data=kernel.symbol, axes=(3, 2, 0, 1)))

    return kernel


@keras_mxnet_symbol
def _preprocess_dw_convnd_kernel(kernel, data_format):
    # transpose depthwise and separable conv2d kernels
    if data_format == 'channels_last':
        if len(kernel.shape) > 3:
            kernel = KerasSymbol(mx.sym.transpose(data=kernel.symbol, axes=(2, 3, 0, 1)))

    return kernel


@keras_mxnet_symbol
def _preprocess_convnd_transpose_output(output_shape, data_format):
    if data_format == 'channels_last':
        output_shape = output_shape[1:-1]
    elif data_format == 'channels_first':
        output_shape = output_shape[2:]
    return output_shape


def _validate_conv_input_shape(input_shape):
    # MXNet convolution operator cannot automatically infer shape.
    # Feature requirement -
    nd = len(input_shape) - 2
    for dim in range(nd):
        if not input_shape[2 + dim]:
            raise ValueError('MXNet Backend: Cannot automatically infer shape for convolution operator.'
                             'Please provide input shape. Given input shape - ', input_shape)


def _calculate_padding_requirement(input_shape, kernel, strides, dilation, border_mode):
    out_size = _calculate_conv_output_size(input_shape, kernel, border_mode, strides, dilation)
    pad_along = dilation * kernel - input_shape - strides - dilation + out_size * strides + 1
    result = int(np.ceil(pad_along / 2.0)), pad_along % 2 != 0, out_size
    return result


def _preprocess_padding_mode(padding_mode, input_shape, kernel, strides, dilation):
    nd = len(input_shape) - 2
    is_slice = (False,) * nd
    out_size = (0,) * nd
    _validate_conv_input_shape(input_shape)
    if padding_mode == 'same' or padding_mode == 'full':
        padding, is_slice, out_size = zip(
            *[_calculate_padding_requirement(input_shape[2 + i], kernel[i],
                                             strides[i], dilation[i], padding_mode)
              for i in range(nd)])
    elif padding_mode == 'valid':
        padding = (0,) * nd
    else:
        raise ValueError('MXNet Backend: Invalid padding mode:', padding_mode)

    return padding, np.any(is_slice), out_size


def _layout_kernel(kernel):
    layout_kernel = tuple(kernel[2:])
    nb_filter = kernel[0]

    return layout_kernel, nb_filter


def _mx_dw_conv(data, num_in_channel, weight=None, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=None,
                depth_mult=1):
    # pure mxnet version of Separable Convolution
    # depthwise convolution
    channels = mx.sym.split(data=data, axis=1, num_outputs=num_in_channel)  # for new version of mxnet > 0.8

    depthwise_outs = [
        mx.sym.Convolution(data=channels[i], num_filter=num_in_channel * depth_mult, kernel=kernel, weight=weight,
                           stride=stride, pad=pad, name=name + 'depthwise_kernel_' + str(i))
        for i in range(num_in_channel)]
    depthwise_out = mx.sym.concat(*depthwise_outs)
    return depthwise_out


def _mx_sp_conv(data, num_in_channel, num_out_channel,
                dw_kernel_shape=(3, 3), dw_kernel_weight=None,
                pw_kernel_shape=(1, 1), pw_kernel_weight=None,
                stride=(1, 1), pad=(1, 1), name=None, depth_mult=1):
    # pure mxnet version of Separable Convolution
    # depthwise convolution
    channels = mx.sym.split(data=data, axis=1, num_outputs=num_in_channel)  # for new version of mxnet > 0.8
    # channels = mx.sym.SliceChannel(data=data, axis=1,
    #                              num_outputs=num_in_channel)  # for old version of mxnet <= 0.8
    depthwise_outs = [mx.sym.Convolution(data=channels[i], num_filter=num_in_channel,
                                         kernel=dw_kernel_shape, weight=dw_kernel_weight,
                                         stride=stride, pad=pad, name=name + 'dw_conv' + str(i))
                      for i in range(num_in_channel)]
    depthwise_out = mx.sym.concat(*depthwise_outs)
    out = mx.sym.Convolution(data=depthwise_out, num_filter=num_out_channel,
                             kernel=pw_kernel_shape, weight=pw_kernel_weight,
                             stride=(1, 1), pad=pad, name='pw_conv')
    return out


@keras_mxnet_symbol
def _sp_convnd(x, depthwise_kernel, pointwise_kernel, strides, filter_dilation, name=None, padding_mode='valid',
               data_format='default'):
    """
    Full math implementation of separable convnd
    """

    if data_format is None or data_format == 'default':
        data_format = image_data_format()

    # Handle Data Format
    x = _preprocess_convnd_input(x, data_format)
    depthwise_kernel = _preprocess_dw_convnd_kernel(depthwise_kernel, data_format)
    pointwise_kernel = _preprocess_dw_convnd_kernel(pointwise_kernel, data_format)

    # We have already converted kernel to match MXNet required shape:
    # (depth, input_depth, rows, cols)
    depthwise_kernel_shape = depthwise_kernel.shape
    num_in_channel = depthwise_kernel_shape[0]
    depthwise_kernel_shape = tuple(depthwise_kernel_shape[2:])

    pointwise_kernel_shape = pointwise_kernel.shape
    num_out_channel = pointwise_kernel_shape[1]
    pointwise_kernel_shape = tuple(pointwise_kernel_shape[2:])

    # Calculate padding requirement.
    padding, is_slice, out_size = _preprocess_padding_mode(padding_mode, x.shape,
                                                           depthwise_kernel, strides,
                                                           filter_dilation)

    # Perform convolution.
    conv = _mx_sp_conv(x.symbol, num_in_channel, num_out_channel,
                       dw_kernel_shape=depthwise_kernel_shape, dw_kernel_weight=depthwise_kernel.symbol,
                       pw_kernel_shape=pointwise_kernel_shape, pw_kernel_weight=pointwise_kernel.symbol,
                       stride=strides, pad=padding,
                       name=_prepare_name(name, "sp_convnd"), depth_mult=1)

    if is_slice:
        begin = (0, 0) + (0,) * len(out_size)
        end = (None, None) + tuple(out_size)
        conv = mx.sym.slice_axis(conv, axis=2, begin=begin[2], end=end[2])
        conv = mx.sym.slice_axis(conv, axis=3, begin=begin[3], end=end[3])

    # Handle original Data Format
    result = _postprocess_convnd_output(KerasSymbol(conv), data_format)
    return result


@keras_mxnet_symbol
def _dw_conv(x, kernel, strides, filter_dilation, name=None, padding_mode='valid',
             data_format='default'):
    if data_format is None or data_format == 'default':
        data_format = image_data_format()

    # Handle Data Format
    x = _preprocess_convnd_input(x, data_format)
    kernel = _preprocess_dw_convnd_kernel(kernel, data_format)

    # We have already converted kernel to match MXNet required shape:
    # (depth, input_depth, rows, cols)
    kernel_shape = kernel.shape
    layout_kernel = tuple(kernel_shape[2:])
    nb_filter = kernel_shape[0]
    depth_multiplier = kernel_shape[1]
    # Calculate padding requirement.
    padding, is_slice, out_size = _preprocess_padding_mode(padding_mode, x.shape,
                                                           layout_kernel, strides,
                                                           filter_dilation)

    # Perform convolution.
    """
    # mathematical implementation
    conv = _mx_dw_conv(x.symbol, nb_filter, weight=kernel.symbol, kernel=layout_kernel, stride=strides, pad=padding,
                        name=_prepare_name(name, "convnd"), depth_mult=1)
    """
    # num_group trick in native conv2d, only support depth_multiplier = 1
    if depth_multiplier != 1:
        raise ValueError('MXNet Backend: Does not support depth multiplier not equal to 1')
    conv = mx.sym.Convolution(data=x.symbol, name=_prepare_name(name, "convnd"),
                              kernel=layout_kernel, num_group=nb_filter, stride=strides, pad=padding,
                              num_filter=nb_filter, weight=kernel.symbol,
                              dilate=filter_dilation, no_bias=True)

    if is_slice:
        begin = (0, 0) + (0,) * len(out_size)
        end = (None, None) + tuple(out_size)
        conv = mx.sym.slice_axis(conv, axis=2, begin=begin[2], end=end[2])
        conv = mx.sym.slice_axis(conv, axis=3, begin=begin[3], end=end[3])

    # Handle original Data Format
    result = _postprocess_convnd_output(KerasSymbol(conv), data_format)
    return result


@keras_mxnet_symbol
def _convnd(x, kernel, strides, filter_dilation, name=None, padding_mode='valid',
            data_format='default'):
    if data_format is None or data_format == 'default':
        data_format = image_data_format()

    if data_format == 'channels_last':
        warnings.warn(
            'MXNet Backend performs best with `channels_first` format. Using '
            '`channels_last` will significantly reduce performance due to the '
            'Transpose operations. '
            'For performance improvement, please use this API'
            '`keras.utils.to_channels_first(x_input)`'
            'to transform `channels_last` data to `channels_first` format and '
            'also please change the `image_data_format` in `keras.json` to '
            '`channels_first`.'
            'Note: `x_input` is a Numpy tensor or a list of Numpy tensor'
            'Refer to: '
            'https://github.com/awslabs/keras-apache-mxnet/tree/master/docs/mxnet_backend/performance_guide.md',
            stacklevel=2)

    # Handle Data Format
    x = _preprocess_convnd_input(x, data_format)
    kernel = _preprocess_convnd_kernel(kernel, data_format)

    # We have already converted kernel to match MXNet required shape:
    # (depth, input_depth, rows, cols)
    kernel_shape = kernel.shape
    layout_kernel = tuple(kernel_shape[2:])
    nb_filter = kernel_shape[0]
    # Calculate padding requirement.
    padding, is_slice, out_size = _preprocess_padding_mode(padding_mode, x.shape,
                                                           layout_kernel, strides,
                                                           filter_dilation)

    # Perform convolution.
    conv = mx.sym.Convolution(data=x.symbol, name=_prepare_name(name, "convnd"),
                              kernel=layout_kernel, stride=strides, pad=padding,
                              num_filter=nb_filter, weight=kernel.symbol,
                              dilate=filter_dilation, no_bias=True)

    if is_slice:
        begin = (0, 0) + (0,) * len(out_size)
        end = (None, None) + tuple(out_size)
        conv = mx.sym.slice_axis(conv, axis=2, begin=begin[2], end=end[2])
        conv = mx.sym.slice_axis(conv, axis=3, begin=begin[3], end=end[3])

    # Handle original Data Format
    result = _postprocess_convnd_output(KerasSymbol(conv), data_format)
    return result


@keras_mxnet_symbol
def _convnd_transpose(x, kernel, output_shape, strides, data_format, name=None, dilation_rate=(1, 1)):
    # Handle Data Format
    x = _preprocess_convnd_input(x, data_format)
    kernel = _preprocess_convnd_kernel(kernel, data_format)

    # We have already converted kernel to match MXNet required shape:
    # (depth, input_depth, rows, cols)
    kernel_shape = kernel.shape
    layout_kernel = tuple(kernel_shape[2:])
    nb_filter = kernel_shape[1]

    # Handle output shape to suit mxnet input format
    if data_format == 'channels_first':
        output_shape = output_shape[2:]
    else:
        output_shape = output_shape[1:-1]

    # Perform transpose convolution
    deconv = mx.sym.Deconvolution(data=x.symbol, name=_prepare_name(name, "convnd_transpose"),
                                  kernel=layout_kernel, stride=strides,
                                  num_filter=nb_filter, weight=kernel.symbol,
                                  no_bias=True, target_shape=output_shape, dilate=dilation_rate)

    # Handle original Data Format
    result = _postprocess_convnd_output(KerasSymbol(deconv), data_format)
    return result


# Pooling helpers
def _calculate_pool_output_size(input_length, filter_size, padding, stride,
                                dilation=1):
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'causal':
        output_length = input_length
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride


def _validate_pool_input_shape(input_shape):
    # MXNet pooling operator cannot automatically infer shape.
    nd = len(input_shape) - 2
    for dim in range(nd):
        if not input_shape[2 + dim]:
            raise ValueError('MXNet Backend: Cannot automatically infer shape for pooling operator.'
                             'Please provide input shape. Given input shape - ', input_shape)


def _calculate_pool_padding_requirement(input_shape, kernel, strides, border_mode, dilation=1):
    out_size = _calculate_pool_output_size(input_shape, kernel, border_mode, strides)
    pad_along = dilation * kernel - input_shape - strides - dilation + out_size * strides + 1
    result = int(np.ceil(pad_along / 2.0)), kernel % 2 == 0, out_size
    return result


def _preprocess_pooling_padding_mode(padding_mode, input_shape, kernel, strides):
    nd = len(input_shape) - 2
    is_slice = (False,) * nd
    out_size = (0,) * nd
    _validate_pool_input_shape(input_shape)
    if padding_mode == 'same':
        padding, is_slice, out_size = zip(
            *[_calculate_pool_padding_requirement(input_shape[2 + i], kernel[i],
                                                  strides[i], padding_mode)
              for i in range(nd)])
    elif padding_mode == 'valid':
        padding = (0,) * nd
    else:
        raise ValueError('MXNet Backend: Invalid padding mode:', padding_mode)

    return padding, np.any(is_slice), out_size


@keras_mxnet_symbol
def _poolnd(x, name, pool_size, strides, padding_mode='valid',
            data_format=None, pool_mode='max'):
    if data_format is None or data_format == 'default':
        data_format = image_data_format()

    _validate_data_format(data_format)
    _validate_pool_mode(pool_mode)
    _validate_padding_mode(padding_mode)

    # Handle Data Format
    x = _preprocess_convnd_input(x, data_format)

    # Calculate padding requirement.
    padding, is_slice, out_size = _preprocess_pooling_padding_mode(padding_mode, x.shape, pool_size, strides)

    if padding_mode == 'same':
        padding_mode = 'valid'
    # Perform Pooling
    mx_out = mx.sym.Pooling(data=x.symbol,
                            name=_prepare_name(name, 'poolnd'),
                            kernel=pool_size,
                            pool_type=pool_mode,
                            pooling_convention=padding_mode,
                            stride=strides, pad=padding)

    if is_slice:
        begin = (0, 0) + (0,) * len(out_size)
        end = (None, None) + tuple(out_size)
        for idx in range(2, len(out_size)):
            mx_out = mx.sym.slice_axis(mx_out, axis=idx, begin=begin[idx], end=end[idx])

    # Handle original Data Format
    result = _postprocess_convnd_output(KerasSymbol(mx_out), data_format)
    return result


def get_mxnet_model_info(model):
    """Get native MXNet model details for given Keras Model.
    `data_names` and `data_shapes` are returned that can be used to bind the model in MXNet Module.
    `data_names` and `data_shapes` represents input layer name and shape.
    You can change the first dimension of data_shapes to match batch size for inference.

    Note: You should use `save_mxnet_model()` API for saving the model in native MXNet model format.

    # Arguments
        model: Keras model instance from which to extract MXNet model details.

    # Returns
        data_names, data_shapes

    # Raises
        AssertionError if Model is not compiled.
    """
    assert model is not None, 'MXNet Backend: Invalid state. Model cannot be None.'

    # Underlying MXNet model for Inference in native MXNet engine.
    symbol = model._pred_mxnet_symbol
    module = model._module

    assert symbol is not None, 'MXNet Backend: Invalid state. MXNet Symbol cannot be None.'
    assert module is not None, 'MXNet Backend: Invalid state. MXNet Module cannot be None.'

    # Get Module Input data_names and data_shapes.
    # This info will be useful for users to easily bind the exported model in MXNet.
    pred_module = module._buckets['pred']
    data_names = pred_module.data_names
    data_shapes = pred_module.data_shapes
    return data_names, data_shapes


def get_num_gpus():
    try:
        gpus = mx.test_utils.list_gpus()
    except CalledProcessError:
        gpus = []
    if gpus and len(gpus) > 0:
        return len(gpus)
    return 0


def _get_mxnet_context(context):
    # If we are currently under a global scope context,
    # then it overrides all other local context parameters.
    global _CURRENT_SCOPE_CTX
    # Note: _CURRENT_SCOPE_CTX will be None if not in scope.
    if _CURRENT_SCOPE_CTX:
        return _CURRENT_SCOPE_CTX

    mxnet_context = []

    if context is None:
        # If user does not provide any context, if GPUs are detected, by default it runs on first available
        # GPU device. If not GPUs are detected, then it falls back to CPU.
        try:
            gpus = mx.test_utils.list_gpus()
        except CalledProcessError:
            gpus = []
        if gpus and len(gpus) > 0:
            mxnet_context.append(mx.gpu(gpus[0]))
        else:
            mxnet_context.append(mx.current_context())
    elif isinstance(context, Number):
        # If user provides number of GPUs to use, set context accordingly.
        if context == 0:
            mxnet_context.append(mx.current_context())
        else:
            for gpu_id in range(0, context):
                mxnet_context.append(mx.gpu(gpu_id))
    elif isinstance(context, str):
        # If user provides GPU context in the format - "gpu(0)" or "eia" or "eia(0)" i.e., string.
        if context.lower().startswith('eia('):
            index = int(context[4:-1])
            mxnet_context.append(mx.eia(index))
        elif context.lower().startswith('gpu('):
            index = int(context[4:-1])
            mxnet_context.append(mx.gpu(index))
        elif context.lower() in mx.Context.devstr2type:
            mxnet_context.append(mx.Context(context.lower()))
        else:
            raise ValueError("Invalid MXNet context provided - ", context)
    else:
        # If user has provided a list.
        # List can be:
        #   1. List of GPU IDs - [0, 1, 2, 3]
        #   2. List of GPU context strings - ["gpu(0)", "gpu(1)"] or ["gpu0", "gpu1"]
        #      Or, ["eia(0)", "eia(1)"]
        for context_name in context:
            if isinstance(context_name, Number):
                mxnet_context.append(mx.gpu(context_name))
            elif context_name.startswith('cpu'):
                mxnet_context.append(mx.cpu())
            elif context_name.startswith('gpu('):
                index = int(context_name[4:-1])
                mxnet_context.append(mx.gpu(index))
            elif context_name.startswith('gpu'):
                index = int(context_name[3:])
                mxnet_context.append(mx.gpu(index))
            elif context_name.startswith('eia('):
                index = int(context_name[4:-1])
                mxnet_context.append(mx.eia(index))
            elif context_name.startswith('eia'):
                index = int(context_name[3:])
                mxnet_context.append(mx.eia(index))
            else:
                raise ValueError("Invalid MXNet context provided - ", context)

    return mxnet_context


class Context:
    """Scope for managing the context/runtime.

    # Example

    ```python
    from keras import backend as K
    from keras.models import load_model

    with K.Context("EIA"):
        model = load_model('imagenet_vgg16.h5')

    prediction = model.predict(input_data)
    ```

    """
    def __init__(self, ctx):
        self.scope_ctx = _get_mxnet_context(ctx)

    def __enter__(self):
        global _CURRENT_SCOPE_CTX
        _CURRENT_SCOPE_CTX = self.scope_ctx

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_SCOPE_CTX
        _CURRENT_SCOPE_CTX = None


def get_model():
    """Prepares Model class that can be used for training a Keras model with MXNet backend.
    Inherits and extends keras.engine.Model class.

    # Returns
        MXNet Model reference
    """
    import importlib
    engine = importlib.import_module('keras.engine.training')

    class Model(engine.Model):
        """The `Model` class adds training & evaluation routines to a `Network`. This class extends
        keras.engine.Model to add MXNet Module to perform training and inference with MXNet backend.
        """

        def __init__(self, *args, **kwargs):
            if 'name' not in kwargs:
                prefix = self.__class__.__name__.lower()
                name = prefix + '_' + str(get_uid(prefix))
                kwargs['name'] = name

            self.name = kwargs['name']

            super(Model, self).__init__(*args, **kwargs)

            if 'context' not in kwargs:
                kwargs['context'] = None

            if 'kvstore' not in kwargs:
                kwargs['kvstore'] = 'device'

            self._context = _get_mxnet_context(kwargs['context'])
            self._kvstore = kwargs['kvstore']

            self._data_names = None
            self._label_names = None
            self._ntrain = None
            self._train_mxnet_symbol = None
            self._train_updates = None
            self._ntest = None
            self._test_mxnet_symbol = None
            self._test_updates = None
            self._npred = None
            self._pred_mxnet_symbol = None
            self._arg_names = None
            self._aux_names = None
            self._fixed_weights = None
            self._args = None
            self._auxs = None
            self._weights_dirty = None
            self._module = None

            self.compiled = False

            if self.built:
                self._num_data = len(self.inputs)
                self._num_label = len(self.outputs) + len(self.output_names)
                # Create Module for Inference
                self._create_predict_module()
            else:
                self._num_data = None
                self._num_label = None

        def compile(self, optimizer, loss=None, metrics=None, loss_weights=None,
                    sample_weight_mode=None, **kwargs):
            super(Model, self).compile(
                optimizer, loss, metrics, loss_weights,
                sample_weight_mode, **kwargs)

            if not self.built:
                # Model is not compilable because
                # it does not know its number of inputs
                # and outputs, nor their shapes and names.
                # We will compile after the first
                # time the model gets called on training data.
                return

            # If context is passed in kwargs
            if 'context' in kwargs:
                self._context = self.get_mxnet_context(kwargs['context'])

            if self.built:
                self._num_data = len(self.inputs)
                self._num_label = len(self.outputs) + len(self.output_names)

            # set the data and label
            self._data_names = [x.name for x in self.inputs if x]
            self._label_names = [x.name for x in self.targets + self.sample_weights if x]

            # set for training
            old = learning_phase()
            set_learning_phase(1)
            self._ntrain = len(self.metrics_tensors) + 1
            train_updates = [stop_gradient(x[1]) for x in self.updates]
            train_keras_symbol = group(
                [make_loss(self.total_loss)] + [stop_gradient(x)
                                                for x in self.metrics_tensors] + train_updates
            )
            bind_values = dfs_get_bind_values(train_keras_symbol)
            self._train_mxnet_symbol = train_keras_symbol.symbol
            symbol_name_map = {i.name: j.name for (_, i), j in zip(self.updates, train_updates)}
            self._train_updates = {dst.name: symbol_name_map[src.name] for dst, src in self.updates}

            # set for testing
            set_learning_phase(0)
            self._ntest = len(self.metrics_tensors) + 1
            state_updates = [x[1] for x in self.state_updates]
            test_keras_symbol = group(
                [self.total_loss] +
                [stop_gradient(x) for x in self.metrics_tensors] +
                state_updates
            )
            bind_values.update(dfs_get_bind_values(test_keras_symbol))
            self._test_mxnet_symbol = test_keras_symbol.symbol

            # set for prediction
            self._npred = len(self.outputs)
            pred_keras_symbol = group(
                self.outputs +
                [symbol for symbol in state_updates if symbol not in self.outputs]
            )
            bind_values.update(dfs_get_bind_values(pred_keras_symbol))
            self._pred_mxnet_symbol = pred_keras_symbol.symbol
            self._test_updates = {dst.name: src.name for dst, src in self.state_updates}
            set_learning_phase(old)

            # set the args and auxs
            inputs_name_set = set(self._data_names + self._label_names)
            self._arg_names = set([x for x in self._train_mxnet_symbol.list_arguments()
                                   if x not in inputs_name_set])
            self._aux_names = set(self._train_mxnet_symbol.list_auxiliary_states())

            trainable_weights = set([x.name for x in self.trainable_weights])
            self._fixed_weights = [x for x in self._arg_names if x not in trainable_weights]
            # self._args = {x: bind_values[x] for x in self._arg_names if x in bind_values}
            self._args = {}
            for x in self._arg_names:
                if x in bind_values:
                    if is_sparse(x):
                        self._args[x] = mx.nd.array(bind_values[x]).tostype('row_sparse')
                    else:
                        self._args[x] = bind_values[x]
            self._auxs = {x: bind_values[x] for x in self._aux_names if x in bind_values}
            self._weights_dirty = False

            if self._context and hasattr(self._context[0], 'device_type') and self._context[0].device_type == 'eia':
                # Only Prediction is Supported with EIA Context
                self._module = mx.mod.Module(self._pred_mxnet_symbol, data_names=self._data_names,
                                             label_names=self._label_names, context=self._context[0],
                                             fixed_param_names=self._fixed_weights)
            else:
                # set the module
                def sym_gen(phase):
                    if phase == 'train':
                        return self._train_mxnet_symbol, self._data_names, self._label_names
                    elif phase == 'test':
                        return self._test_mxnet_symbol, self._data_names, self._label_names
                    else:
                        return self._pred_mxnet_symbol, self._data_names, None

                self._module = mx.mod.BucketingModule(
                    sym_gen=sym_gen,
                    default_bucket_key='pred',
                    context=self._context,
                    fixed_param_names=self._fixed_weights)

            set_model(self)
            self.compiled = True

        def _adjust_module(self, inputs, phase):
            if not self._module:
                raise RuntimeError('MXNet Backend: You must compile your model before using it.')
            if self._num_data + self._num_label == len(inputs) - 1:
                inputs = inputs[:-1]
            elif self._num_data == len(inputs) - 1:
                inputs = inputs[:-1]
            assert self._num_data == len(inputs) or self._num_data + self._num_label == len(inputs)
            data = [mx.nd.array(x, dtype=s.dtype)
                    for (s, x) in zip(self.inputs, inputs[:self._num_data])]
            data_shapes = [mx.io.DataDesc(s.name, arr.shape, dtype=s.dtype)
                           for (s, arr) in zip(self.inputs, data)]
            if self._num_data < len(inputs):
                label = [mx.nd.array(x, dtype=s.dtype)
                         for (s, x) in zip(self.targets + self.sample_weights,
                                           inputs[self._num_data:])]
                label_shapes = [mx.io.DataDesc(s.name, arr.shape, dtype=s.dtype)
                                for (s, arr) in zip(self.targets + self.sample_weights, label)]
            else:
                label = None
                label_shapes = None

            if not self._module.binded:
                # allow prediction without compiling the model using different binding
                if phase == 'pred' and not self.compiled:
                    self._module.bind(data_shapes=data_shapes, label_shapes=None,
                                      for_training=False)
                    self._set_weights()
                else:
                    self._module.bind(data_shapes=data_shapes, label_shapes=None, for_training=True)
                    self._set_weights()
                    self._module.init_optimizer(kvstore=self._kvstore, optimizer=self.optimizer)

            # If context is EIA, we will be directly using Module rather than Bucketing Module.
            # Hence, below specialization.
            if isinstance(self._module, mx.mod.BucketingModule):
                self._module.switch_bucket(phase, data_shapes, label_shapes)

                # adjust module data shape
                if inputs[0].shape[0] != self._module._curr_module._exec_group.batch_size:
                    self._module._curr_module.reshape(data_shapes, label_shapes)
                    assert inputs[0].shape[0] == self._module._curr_module._exec_group.batch_size, \
                        'Reshape failed'
            else:
                # adjust module data shape
                if inputs[0].shape[0] != self._module._exec_group.batch_size:
                    self._module.reshape(data_shapes, label_shapes)
                    assert inputs[0].shape[0] == self._module._exec_group.batch_size, \
                        'Reshape failed'

            return data, label, phase, data_shapes, label_shapes

        def _sync_weights(self):
            if self._weights_dirty:
                args, auxs = self._module.get_params()
                for name in self._arg_names:
                    try:
                        self._args[name][:] = args[name]
                    except:
                        # when name is not in self._args (key not found)
                        self._args[name] = []
                        self._args[name][:] = args[name]
                for name in self._aux_names:
                    try:
                        self._auxs[name][:] = auxs[name]
                    except:
                        # when name is not in self._auxs (key not found)
                        self._auxs[name] = []
                        self._auxs[name][:] = auxs[name]
                self._weights_dirty = False

        def _set_weights(self, arg_params=None, auxs_params=None):
            if self._module.binded:
                self._module.set_params(self._args if arg_params is None else arg_params,
                                        self._auxs if auxs_params is None else auxs_params,
                                        allow_missing=True)
                self._weights_dirty = arg_params is not None or auxs_params is not None
            else:
                if arg_params:
                    for k in arg_params:
                        self._args[k][:] = arg_params[k]
                if auxs_params:
                    for k in auxs_params:
                        self._auxs[k][:] = auxs_params[k]
                self._weights_dirty = False

        def _update(self, updates):
            for exe in self._module._curr_module._exec_group.execs:
                outs = exe.output_dict
                args = exe.arg_dict
                for dst, src in updates.items():
                    args[dst][:] = outs[src + '_output']

        def _make_train_function(self):
            def train_function(inputs):
                self._check_trainable_weights_consistency()
                data, label, _, data_shapes, label_shapes = self._adjust_module(inputs, 'train')

                batch = mx.io.DataBatch(data=data, label=label, bucket_key='train',
                                        provide_data=data_shapes, provide_label=label_shapes)

                self._module.forward_backward(batch)
                self._module.update()
                self._update(self._train_updates)
                self._weights_dirty = True
                outs = self._module.get_outputs()[:self._ntrain]
                return [x.asnumpy().mean() for x in outs]

            # If context is EIA this should not be supported
            if self._context and hasattr(self._context[0], 'device_type') and self._context[0].device_type == 'eia':
                raise RuntimeError('MXNet Backend: Model training is not supported with MXNet EIA context.'
                                   'Use CPU/GPU.')

            self.train_function = train_function

        def _make_test_function(self):
            def test_function(inputs):
                # although this function do testing we need the training symbol
                data, label, _, data_shapes, label_shapes = self._adjust_module(inputs, 'test')

                batch = mx.io.DataBatch(data=data, label=label, bucket_key='test',
                                        provide_data=data_shapes, provide_label=label_shapes)
                self._module.forward(batch, is_train=False)
                if self._test_updates:
                    self._update(self._test_updates)
                    self._weights_dirty = True
                outs = self._module.get_outputs()[:self._ntrain]
                return [x.asnumpy().mean() for x in outs]

            # If context is EIA this should not be supported
            if self._context and hasattr(self._context[0], 'device_type') and self._context[0].device_type == 'eia':
                raise RuntimeError(
                    'MXNet Backend: Model Testing is not supported with MXNet EIA context. Use CPU/GPU.')
            self.test_function = test_function

        def _make_predict_function(self):
            def predict_function(inputs):
                # used predict only module if predict is called without compile
                if not self.compiled:
                    if self.built:
                        self._num_data = len(self.inputs)
                        self._num_label = len(self.outputs) + len(self.output_names)
                        # Create Module for Inference
                        self._create_predict_module()
                    self._module = self._predict_only_module
                    set_model(self)

                data, label, _, data_shapes, label_shapes = self._adjust_module(inputs, 'pred')
                batch = mx.io.DataBatch(data=data, label=label, bucket_key='pred',
                                        provide_data=data_shapes, provide_label=label_shapes)
                self._module.forward(batch, is_train=False)
                if self._test_updates:
                    self._update(self._test_updates)
                    self._weights_dirty = True
                outs = self._module.get_outputs()[:self._npred]
                return [x.asnumpy() for x in outs]

            self.predict_function = predict_function

        def _create_predict_module(self):
            # set the data and label
            self._data_names = [x.name for x in self.inputs if x]

            state_updates = [x[1] for x in self.state_updates]
            # set for prediction
            self._npred = len(self.outputs)
            pred_keras_symbol = group(
                self.outputs +
                [symbol for symbol in state_updates if symbol not in self.outputs]
            )
            bind_values = dfs_get_bind_values(pred_keras_symbol)
            self._pred_mxnet_symbol = pred_keras_symbol.symbol

            # set the args and auxs
            inputs_name_set = set(self._data_names)
            self._arg_names = set([x for x in self._pred_mxnet_symbol.list_arguments()
                                   if x not in inputs_name_set])
            self._aux_names = set(self._pred_mxnet_symbol.list_auxiliary_states())

            trainable_weights = set([x.name for x in self.trainable_weights])
            self._fixed_weights = [x for x in self._arg_names if x not in trainable_weights]
            self._args = {x: bind_values[x] for x in self._arg_names if x in bind_values}
            self._auxs = {x: bind_values[x] for x in self._aux_names if x in bind_values}
            self._weights_dirty = False

            # set module for prediction only
            if self._context and hasattr(self._context[0], 'device_type') and self._context[0].device_type == 'eia':
                # Only Prediction is Supported with EI Context
                self._predict_only_module = mx.mod.Module(self._pred_mxnet_symbol, data_names=self._data_names,
                                                          label_names=self._label_names, context=self._context[0],
                                                          fixed_param_names=self._fixed_weights)
            else:
                def sym_gen(phase):
                    return self._pred_mxnet_symbol, self._data_names, None

                # separate module for using predict without compiling model
                self._predict_only_module = mx.mod.BucketingModule(
                    sym_gen=sym_gen,
                    default_bucket_key='pred',
                    context=self._context,
                    fixed_param_names=self._fixed_weights)

        def set_mxnet_context(self, context):
            """Sets the mxnet context for the current Model.

            # Arguments
                context: Integer >= 2 or list of integers, number of GPUs or
                      list of GPU IDs on which to create model replicas.
            """
            if isinstance(context, (list, tuple)):
                if len(context) <= 1:
                    raise ValueError('MXNet Backend: For multi-gpu usage to be effective, '
                                     'call `multi_gpu_model` with `len(gpus) >= 2`. '
                                     'Received: `gpus=%s`' % context)
            elif isinstance(context, str):
                if not context.lower().startswith('eia'):
                    raise ValueError('MXNet Backend: Invalid context provided - %s' % context)
            else:
                if context <= 1:
                    raise ValueError('MXNet Backend: For multi-gpu usage to be effective, '
                                     'call `multi_gpu_model` with `gpus >= 2`. '
                                     'Received: `gpus=%d`' % context)

            self._context = _get_mxnet_context(context)

    return Model


def get_sequential_model():
    """Prepares Sequential Model class that can be used for training a Keras Sequential Model with MXNet backend.
    Inherits and extends keras.engine.Model and keras.engine.Sequential class.

    # Returns
        MXNet Sequential Model reference
    """
    import importlib
    sequential = importlib.import_module('keras.engine.sequential')
    engine = importlib.import_module('keras.engine.training')

    class Sequential(sequential.Sequential, engine.Model):
        """Linear stack of layers. This class extends keras.engine.Sequential to add MXNet Module to perform training
        and inference with MXNet backend.
        """

        def __init__(self, layers=None, *args, **kwargs):
            if 'name' not in kwargs:
                prefix = self.__class__.__name__.lower()
                name = prefix + '_' + str(get_uid(prefix))
                kwargs['name'] = name

            self.name = kwargs['name']
            engine.Model.__init__(self, *args, **kwargs)
            self._build_input_shape = None

            # Add to the model any layers passed to the constructor.
            if layers:
                for layer in layers:
                    self.add(layer)

    return Sequential


def get_optimizers():
    import importlib
    optimizers = importlib.import_module('keras.optimizers')

    class MXOptimizer(optimizers.Optimizer, mx.optimizer.Optimizer):
        """Custom MXNet Optimizer wrapping Keras Optimizer.
        This is required because we cannot use Keras optimizer directly as MXNet backend does not
        support symbolic optimizers.
        """

        def __init__(self, lr, decay):
            super(MXOptimizer, self).__init__()
            self.lr = variable(lr)
            self.decay = variable(decay)

        def _get_lr(self, _):
            return self.lr.tensor.asscalar() / (1. + self.decay.tensor.asscalar() * self.num_update)

        def _get_lrs(self, _):
            return [self._get_lr(_) for i in range(0, self.aggregate_num)]

        def get_config(self):
            config = {}
            if hasattr(self, 'clipnorm'):
                config['clipnorm'] = self.clipnorm
            if hasattr(self, 'clipvalue'):
                config['clipvalue'] = self.clipvalue
            return config

    class SGD(MXOptimizer, mx.optimizer.SGD):
        """Stochastic gradient descent optimizer.

        Includes support for momentum,
        learning rate decay, and Nesterov momentum.

        # Arguments
            lr: float >= 0. Learning rate.
            momentum: float >= 0. Parameter that accelerates SGD
                in the relevant direction and dampens oscillations.
            decay: float >= 0. Learning rate decay over each update.
            nesterov: boolean. Whether to apply Nesterov momentum.
        """

        def __init__(self, lr=0.01, momentum=0., decay=0.,
                     nesterov=False, clipnorm=None, **kwargs):
            mx.optimizer.SGD.__init__(self, learning_rate=lr, momentum=momentum, clip_gradient=clipnorm, **kwargs)
            MXOptimizer.__init__(self, lr, decay)
            # use only 1 aggregated optimizer to fit into Keras optimizer requirement
            self.aggregate_num = 1

        def get_config(self):
            config = {'lr': float(get_value(self.lr)),
                      'momentum': float(get_value(self.momentum)),
                      'decay': float(get_value(self.decay))}
            base_config = super(SGD, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    class Adagrad(MXOptimizer, mx.optimizer.AdaGrad):
        """Adagrad optimizer.

        Adagrad is an optimizer with parameter-specific learning rates,
        which are adapted relative to how frequently a parameter gets
        updated during training. The more updates a parameter receives,
        the smaller the updates.

        It is recommended to leave the parameters of this optimizer
        at their default values.

        # Arguments
            lr: float >= 0. Initial learning rate.
            epsilon: float >= 0. If `None`, defaults to `K.epsilon()`.
            decay: float >= 0. Learning rate decay over each update.

        # References
            - [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)  # nopep8
        """

        def __init__(self, lr=0.01, epsilon=1e-8, decay=0., clipnorm=None, **kwargs):
            mx.optimizer.AdaGrad.__init__(self, learning_rate=lr, eps=epsilon, clip_gradient=clipnorm, **kwargs)
            MXOptimizer.__init__(self, lr, decay)

        def get_config(self):
            config = {'lr': float(get_value(self.lr)),
                      'decay': float(get_value(self.decay)),
                      'epsilon': float(get_value(self.float_stable_eps))}
            base_config = super(Adagrad, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    class Adadelta(MXOptimizer, mx.optimizer.AdaDelta):
        """Adadelta optimizer.

        Adadelta is a more robust extension of Adagrad
        that adapts learning rates based on a moving window of gradient updates,
        instead of accumulating all past gradients. This way, Adadelta continues
        learning even when many updates have been done. Compared to Adagrad, in the
        original version of Adadelta you don't have to set an initial learning
        rate. In this version, initial learning rate and decay factor can
        be set, as in most other Keras optimizers.

        It is recommended to leave the parameters of this optimizer
        at their default values.

        # Arguments
            lr: float >= 0. Initial learning rate, defaults to 1.
                It is recommended to leave it at the default value.
            rho: float >= 0. Adadelta decay factor, corresponding to fraction of
                gradient to keep at each time step.
            epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
            decay: float >= 0. Initial learning rate decay.

        # References
            - [Adadelta - an adaptive learning rate method](http://arxiv.org/abs/1212.5701)
        """

        def __init__(self, lr=1.0, rho=0.95, epsilon=1e-8, decay=0., clipnorm=None, **kwargs):
            mx.optimizer.AdaDelta.__init__(self, rho=rho, epsilon=epsilon, clip_gradient=clipnorm, **kwargs)
            MXOptimizer.__init__(self, lr, decay)

        def get_config(self):
            config = {'lr': float(get_value(self.lr)),
                      'rho': float(get_value(self.rho)),
                      'decay': float(get_value(self.decay)),
                      'epsilon': self.epsilon}
            base_config = super(Adadelta, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    class Adam(MXOptimizer, mx.optimizer.Adam):
        """Adam optimizer.

        Default parameters follow those provided in the original paper.

        # Arguments
            lr: float >= 0. Learning rate.
            beta_1: float, 0 < beta < 1. Generally close to 1.
            beta_2: float, 0 < beta < 1. Generally close to 1.
            epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
            decay: float >= 0. Learning rate decay over each update.
            amsgrad: boolean. Whether to apply the AMSGrad variant of this
                algorithm from the paper "On the Convergence of Adam and
                Beyond".

        # References
            - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
            - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
        """

        def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                     epsilon=1e-8, decay=0., clipnorm=None, **kwargs):
            mx.optimizer.Adam.__init__(self, learning_rate=lr, beta1=beta_1, beta2=beta_2,
                                       epsilon=epsilon, clip_gradient=clipnorm, **kwargs)
            MXOptimizer.__init__(self, lr, decay)

        def get_config(self):
            config = {'lr': float(get_value(self.lr)),
                      'beta_1': float(get_value(self.beta1)),
                      'beta_2': float(get_value(self.beta2)),
                      'decay': float(get_value(self.decay)),
                      'epsilon': self.epsilon}
            base_config = super(Adam, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    class Adamax(MXOptimizer, mx.optimizer.Adamax):
        """Adamax optimizer from Adam paper's Section 7.

        It is a variant of Adam based on the infinity norm.
        Default parameters follow those provided in the paper.

        # Arguments
            lr: float >= 0. Learning rate.
            beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
            epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
            decay: float >= 0. Learning rate decay over each update.

        # References
            - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        """

        def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999, decay=0., clipnorm=None,
                     epsilon=1e-8, **kwargs):
            mx.optimizer.Adamax.__init__(self, learning_rate=lr, beta1=beta_1, beta2=beta_2,
                                         clip_gradient=clipnorm, **kwargs)
            MXOptimizer.__init__(self, lr, decay)
            self.epsilon = epsilon

        def get_config(self):
            config = {'lr': float(get_value(self.learning_rate)),
                      'beta_1': float(get_value(self.beta1)),
                      'beta_2': float(get_value(self.beta2)),
                      'decay': float(get_value(self.decay)),
                      'epsilon': self.epsilon}
            base_config = super(Adamax, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    class Nadam(MXOptimizer, mx.optimizer.Nadam):
        """Nesterov Adam optimizer.

        Much like Adam is essentially RMSprop with momentum,
        Nadam is Adam RMSprop with Nesterov momentum.

        Default parameters follow those provided in the paper.
        It is recommended to leave the parameters of this optimizer
        at their default values.

        # Arguments
            lr: float >= 0. Learning rate.
            beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
            epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.

        # References
            - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
            - [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)  # nopep8
        """

        def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0., clipnorm=None,
                     schedule_decay=0.004, **kwargs):
            mx.optimizer.Nadam.__init__(self, learning_rate=lr, beta1=beta_1, beta2=beta_2, epsilon=epsilon,
                                        schedule_decay=schedule_decay, **kwargs)
            MXOptimizer.__init__(self, lr, decay)

        def get_config(self):
            config = {'lr': float(get_value(self.learning_rate)),
                      'beta_1': float(get_value(self.beta1)),
                      'beta_2': float(get_value(self.beta2)),
                      'epsilon': self.epsilon,
                      'schedule_decay': self.schedule_decay}
            base_config = super(Nadam, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    class RMSprop(MXOptimizer, mx.optimizer.RMSProp):
        """RMSProp optimizer.

        It is recommended to leave the parameters of this optimizer
        at their default values
        (except the learning rate, which can be freely tuned).

        This optimizer is usually a good choice for recurrent
        neural networks.

        # Arguments
            lr: float >= 0. Learning rate.
            rho: float >= 0.
            epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
            decay: float >= 0. Learning rate decay over each update.

        # References
            - [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)  # nopep8
        """

        def __init__(self, lr=0.001, rho=0.9, epsilon=1e-8, decay=0., clipnorm=None, **kwargs):
            mx.optimizer.RMSProp.__init__(self, learning_rate=lr, gamma1=rho, epsilon=epsilon,
                                          clip_gradient=clipnorm, **kwargs)
            MXOptimizer.__init__(self, lr, decay)

        def get_config(self):
            config = {'lr': float(get_value(self.lr)),
                      'rho': float(get_value(self.gamma1)),
                      'decay': float(get_value(self.decay)),
                      'epsilon': self.epsilon}
            base_config = super(RMSprop, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return SGD, Adagrad, Adadelta, Adam, Adamax, RMSprop, Nadam
