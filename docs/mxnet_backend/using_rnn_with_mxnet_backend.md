# Using Recurrent Neural Network (RNN)

## Table of Contents

1. [Overview](#overview)
2. [Variable length inputs are not supported for unrolling](#variable-length-inputs-are-not-supported-for-unrolling)
3. [Using Unrolling in RNN](#using-unrolling-in-rnn)
4. [Slower CPU training performance](#slower-cpu-training-performance)

## Overview

In this document, we describe the limitations of using RNNs with MXNet backend and available workarounds for the same.

## Variable length inputs are not supported for unrolling

MXNet backend does not support variable length inputs when you unrolling RNN cells in the recurrent layers. To overcome this limitation, you can
pad the input sequences to prepare fixed length inputs. The MXNet backend requires both the `input_shape` and 
`unroll=True` parameters while adding the SimpleRNN/LSTM/GRU layer.


### Transform variable length to fixed length inputs

You should pad the variable length input sequences to make it a fixed length. You can use Keras API - `keras.preprocessing.sequence.pad_sequences` for padding the input.
 
Below is an example use case:

```python

# Convert variable length input to fixed length by padding.
# Usually, you choose maxlen to be maximum length of the variable input sequence.
# This converts all input to be of length maxlen.
new_x_train = keras.preprocessing.sequence.pad_sequences(old_x_train, maxlen=100)

# Build the Model
print('Build model...')
model = Sequential()
# len(chars) => Feature Size
model.add(LSTM(128, input_shape=(maxlen, len(chars)), unroll=True))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Train the Model
model.fit(new_x_train, y_train, batch_size=128, epochs=60)

```

```
NOTE:
    Padding the input has performance implication due to wasted computation on paddings. You should not pad to 
    arbitrary large maxlen. It is always optimal to choose maxlen for padding to be equal to the max length of the 
    input sequences.
```
## Using unrolling in RNN

In Keras RNN layers, by default unroll is set to False, and it requires control flow operators(e.g. foreach, while_loop).
We recently added support for this feature so by default RNN layers are not unrolled, same as other backends.

Unrolling RNN Cells will have better performance but consumes more memory. It's only suitable for short sequences. For more details, refer to
[Keras RNN API](https://keras.io/layers/recurrent/)

## Slower CPU training performance

Performance of training a model with RNN layers on a CPU with MXNet backend is not optimal. This is a known issue and actively being worked on. Please expect this issue to be resolved in further releases of keras-mxnet. See [benchmark results](../../benchmark/README.md) for more detailed analysis.

```
NOTE:
    There is no performance degradation on GPUs.
```
