'''
Adapted from a simple CNN example for text classification from Keras:
https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py
#This example demonstrates the use of Convolution1D for text classification.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from ml_pipeline import utils
import pandas as pd
import tensorflow  # backend used by keras
import numpy as np


# set parameters:
max_features = 5000
maxlen = 40
batch_size = 32
embedding_dims = 100
filters = 500
kernel_size = 18
hidden_dims = 500
epochs = 12


class CNN_multi:

    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor
        self.model = None

    def encode(self, train_X, train_y, test_X, test_y):
        if self.preprocessor is not None:
            train_X = self.preprocessor.fit_transform(train_X)
            test_X = self.preprocessor.transform(test_X)
        return encode(train_X, train_y, test_X, test_y)

    def fit(self, train_X, train_y):
        self.model = build_model(train_X, train_y)

    def predict(self, test_X):
        #sys_y = (self.model.predict(test_X) > 0.5).astype('int32')
        sys_y = self.model.predict(test_X, batch_size=32)
        sys_y = np.argmax(sys_y, axis=1)
        return sys_y


def encode(train_X, train_y, test_X, test_y):
    # map string labels to integers
    data_y = []
    data_y.extend(train_y)
    data_y.extend(test_y)
    labels = list(set(data_y))
    data_y = [labels.index(y) for y in data_y]
    train_y = data_y[:len(train_y)]
    test_y = data_y[len(train_y):]

    # integer encode the documents
    data_X = []
    data_X.extend(train_X)
    data_X.extend(test_X)

    vocab_size = max_features
    encoded_docs = [one_hot(d, vocab_size) for d in data_X]
    train_X = encoded_docs[:len(train_X)]
    test_X = encoded_docs[len(train_X):]
    print(len(train_X), 'train sequences')
    print(len(test_X), 'data sequences')

    print('Pad sequences (samples x time)')
    train_X = sequence.pad_sequences(train_X, maxlen=maxlen)
    test_X = sequence.pad_sequences(test_X, maxlen=maxlen)
    print('train_X shape:', train_X.shape)
    print('test_X shape:', test_X.shape)
    return pd.DataFrame(train_X), pd.DataFrame(train_y), pd.DataFrame(test_X), pd.DataFrame(test_y)


def encode_data(data_dir):
    print('Loading data...')
    task = of.Offenseval()
    task.load(data_dir=data_dir)
    train_X, train_y, test_X, test_y = utils.get_instances(task, split_train_dev=False)
    print(len(train_X), 'train sequences')
    print(len(test_X), 'data sequences')

    train_X, train_y, test_X, test_y = encode(train_X, train_y, test_X, test_y)

    return train_X, train_y, test_X, test_y


def build_model(train_X, train_y):
    print('Build model...')
    tensorflow.keras.backend.clear_session()
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_X, train_y,
              batch_size=batch_size,
              epochs=epochs)

    return model


def evaluate_multi(model, test_X, test_y):
    sys_y = model.predict(test_X, batch_size=32)
    sys_y = np.argmax(sys_y, axis=1)
    #sys_y = (model.predict(test_X) > 0.5).astype('int32')
    #sys_y = model.predict(test_X > 0.5).astype('int32')
    print(utils.eval(test_y, sys_y))


def build_and_evaluate_model(data_dir):
    train_X, train_y, test_X, test_y = encode_data(data_dir)
    model = build_model(train_X, train_y)
    evaluate(model, test_X, test_y)


def load_and_evaluate_model(model_dir):
    train_X, train_y, test_X, test_y = utils.load_data(model_dir)
    model = utils.load_pretrained_model(model_dir)
    evaluate(model, test_X, test_y)
