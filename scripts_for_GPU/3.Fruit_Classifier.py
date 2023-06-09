﻿

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AvgPool2D, Flatten, Dense
import numpy as np


def load_train(path):
    features_train = np.load(path + 'train_features.npy')
    target_train = np.load(path + 'train_target.npy')
    features_train = features_train.reshape(features_train.shape[0], 28, 28, 1) / 255 
    return features_train, target_train

def load_test(path):
    features_test = np.load(path + 'test_features.npy')
    target_test = np.load(path + 'test_target.npy')
    features_test = features_test.reshape(features_test.shape[0], 28, 28, 1) / 255
    return features_test, target_test

def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation="tanh", input_shape=(28, 28, 1)))  # Свёрточные слой 1
    model.add(AvgPool2D(pool_size=(2, 2))) # Пуллинг 1
    model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation="tanh")) # Свёрточные слой 2
    model.add(AvgPool2D(pool_size=(2, 2))) # Пуллинг 2
    model.add(Flatten())
    model.add(Dense(units=120, activation='tanh')) # Полносвязный слой 1
    model.add(Dense(units=84, activation='tanh'))  # Полносвязный слой 2
    model.add(Dense(units=10, activation='softmax'))  # Полносвязный слой 3
    optimizer = Adam(lr=0.0003)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc']) 
    return model

def train_model(model, train_data, test_data,
                batch_size=32, epochs=10,
                steps_per_epoch=None, validation_steps=None):

    features_train, target_train = train_data
    features_test, target_test = test_data
    model.fit(features_train, target_train, 
              validation_data=(features_test, target_test),
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)
    return model