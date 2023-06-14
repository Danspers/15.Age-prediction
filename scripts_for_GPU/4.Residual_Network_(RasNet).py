

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_train(path):
    # создание загрузчика данных
    train_datagen = ImageDataGenerator(rescale=1/255., vertical_flip=True, horizontal_flip=True)
    
    # загрузка данных (по частям/батчам)
    train_datagen_flow = train_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        seed=12345)
    
    return train_datagen_flow


def create_model(input_shape):
    backbone = ResNet50(input_shape=input_shape,
                        weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        include_top=False)
    
    model = Sequential() # инициализация многослойной структуры
    model.add(backbone) # Средний слой - Костяк из внутренних повторяющихся блоков свёртки
    model.add(GlobalAveragePooling2D()) # Ending - Глобальный пуллинг (по всей картинке)
    model.add(Dense(12, activation='softmax')) # Ending - Полносвязный слой

    optimizer = Adam(lr=0.0002)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc']) 
    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=3, steps_per_epoch=None, validation_steps=None):
    model.fit(train_data, 
              validation_data=test_data,
              batch_size=batch_size,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)

    return model