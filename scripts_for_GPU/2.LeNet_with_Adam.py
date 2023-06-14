

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AvgPool2D, Flatten, Dense
import numpy as np


def load_train(path):
    # создание загрузчика данных
    train_datagen = ImageDataGenerator(
        rescale=1/255.,
        vertical_flip=True,
        horizontal_flip=True,
        width_shift_range=0.10,
        height_shift_range=0.10)
    
    # загрузка данных (по частям/батчам)
    train_datagen_flow = train_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        seed=12345)
    
    return train_datagen_flow
    

def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation="relu", input_shape=input_shape))  # Свёрточные слой 1
    model.add(AvgPool2D(pool_size=(2, 2))) # Пуллинг 1
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu")) # Свёрточные слой 2
    model.add(AvgPool2D(pool_size=(2, 2))) # Пуллинг 2
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))  # Свёрточные слой 3
    model.add(AvgPool2D(pool_size=(2, 2))) # Пуллинг
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu')) # Полносвязный слой 1
    model.add(Dense(units=64, activation='relu'))  # Полносвязный слой 2
    model.add(Dense(units=12, activation='softmax'))  # Полносвязный слой 3

    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc']) 
    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=10, steps_per_epoch=None, validation_steps=None):
    model.fit(train_data, 
              validation_data=test_data,
              batch_size=batch_size,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)

    return model