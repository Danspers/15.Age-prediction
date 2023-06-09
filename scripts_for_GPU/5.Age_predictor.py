﻿

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_train(path):
    # создание загрузчика данных
    train_datagen = ImageDataGenerator(validation_split=0.25, rescale=1/255) #, vertical_flip=True, horizontal_flip=True)
    
    # загрузка данных (по частям/батчам)
    train_datagen_flow = train_datagen.flow_from_dataframe(
        directory=path + '/final_files',
        dataframe=pd.read_csv(path + '/labels.csv'),
        x_col='file_name',
        y_col='real_age',
        target_size=(200, 200),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=12345)
    
    return train_datagen_flow

def load_test(path):
    valid_datagen = ImageDataGenerator(validation_split=0.25, rescale=1/255)
    
    valid_datagen_flow = valid_datagen.flow_from_dataframe(
        directory=path + '/final_files',
        dataframe=pd.read_csv(path + '/labels.csv'),
        x_col='file_name',
        y_col='real_age',
        target_size=(200, 200),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=12345)
    
    return valid_datagen_flow

def create_model(input_shape):
    backbone = ResNet50(input_shape=input_shape,
                        weights='imagenet', #/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
                        include_top=False)
    
    model = Sequential() # инициализация многослойной структуры
    model.add(backbone) # Средний слой - Костяк из внутренних повторяющихся блоков свёртки
    model.add(GlobalAveragePooling2D()) # Ending - Глобальный пуллинг (по всей картинке)
    model.add(Dense(1, activation='relu')) # Ending - Полносвязный слой

    optimizer = Adam(lr=0.0005)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae']) 
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