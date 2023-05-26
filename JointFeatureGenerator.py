import random

import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Concatenate, Dense, Dropout, LSTM, Reshape, Conv2D, MaxPooling2D, TimeDistributed, \
    Flatten, Lambda, Layer, BatchNormalization


def create_joint_feature_generator(input_size_eme,
                                   output_size=128, lstm_units=128, hidden_units=256, dropout_rate=0.1):
    eme_input_shape = input_size_eme
    eme_input = Input(shape=eme_input_shape, name='eme_input')

    reshape_input = Reshape((240, 80, 80))(eme_input)

    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(reshape_input)
    max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(max_pool2)
    max_pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    frame_level_flatten = TimeDistributed(Flatten())(max_pool3)
    frame_level_lstm = LSTM(lstm_units, return_sequences=True)(frame_level_flatten)
    sequence_level_lstm = LSTM(lstm_units, return_sequences=False)(frame_level_lstm)

    fc1 = Dense(hidden_units, activation='relu')(sequence_level_lstm)
    norm1 = BatchNormalization(momentum=0.9)(fc1)
    # dropout1 = Dropout(dropout_rate)(fc1)
    fc2 = Dense(hidden_units, activation='relu')(norm1)
    norm2 = BatchNormalization(momentum=0.9)(fc2)
    # dropout2 = Dropout(dropout_rate)(fc2)
    joint_feature = Dense(output_size, activation='relu')(norm2)

    jfg_model = Model(inputs=eme_input, outputs=joint_feature)

    return jfg_model


def create_jfg_classifier(jfg_model, num_classes):
    user_class = Dense(num_classes, activation='softmax')(jfg_model.output)
    model = Model(inputs=jfg_model.input, outputs=user_class)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
