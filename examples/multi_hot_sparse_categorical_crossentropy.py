'''Trains a simple convnet on multi label classification using multi_hot_sparse_categorical_crossentropy

This example demonstrate
1) how to do multi label classification using normal categorical crossentropy
2) when labels are sparse, how to improve performance using multi_hot_sparse_categorical_crossentropy
'''

import time
import random
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.sequence import pad_sequences

"""
Input:
input data is random images of size (32, 32) in channels first data format

Labels:
Tradition labels are in the shape of (num_samples, num_class), for example:
labels = [[0, 1, 1, ..., 0],
          [1, 1, 0, ..., 0],
          ...
          [0, 0, 0, ..., 1]]
where len(labels) = num_samples, len(labels[0]) = num_classes

However, when num_classes are very large and labels are very sparse,
we can represent them differently, for example:

There are total 1000 classes, so there will be 10000 different labels.
Each image can belong to at most 5 labels at the same time.
labels = [[1, 2],
          [0, 1],
          ...
          [999]]
where labels is a list of list

Special Note:
To deal with different length of sparse labels, we pad them with negative values,
so we can differentiate padding values with normal labels. It will become:
padded_labels = pad_sequeences(labels, value=-1)
padded_labels = [[-1, -1, -1, 1, 2],
          [-1, -1, -1, 0, 1],
          ...
          [-1, -1, -1, -1, 999]]
It will have shape (num_samples, 5) which still save space compare to dense labels.
"""

# input image dimensions
img_rows, img_cols = 28, 28
epoch = 5
num_gpus = 4
num_classes = 1000
num_samples = 50000
input_shape = (3, img_rows, img_cols)

# creating random images of size (28, 28) as training data
x_train = np.random.randint(0, 256, (num_samples, 3, img_rows, img_cols))

# creating dense labels and sparse labels
sparse_labels = []
dense_labels = np.zeros((num_samples, num_classes))
for i in range(0, num_samples):
    # each data have at most 5 labels
    for j in range(0, 5):
        label = random.randint(0, num_classes - 1)
        dense_labels[i][label] = 1
        # making the number of labels for each data unequal
        if random.randint(0, 5) == 1:
            break
    sparse_label_j = np.where(dense_labels[i] == 1)[0]
    sparse_labels.append(sparse_label_j)


# construct a simple CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# use normal categorical crossentropy
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
start = time.time()
model.fit(x_train, dense_labels,
          batch_size=32,
          epochs=epoch,
          verbose=1)
print("categorical_crossentropy time:", time.time() - start)


# use normal multi_hot_sparse_categorical_crossentropy
model.compile(loss=keras.losses.multi_hot_sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# pad sparse labels into shape length with value -1 to differentiate from normal labels
y_train_pad = pad_sequences(sparse_labels, value=-1)

start = time.time()
model.fit(x_train, y_train_pad,
          batch_size=32,
          epochs=epoch,
          verbose=1)
print("multi_hot_categorical_crossentropy time:", time.time() - start)
