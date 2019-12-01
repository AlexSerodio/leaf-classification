from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

import prepare_dataset as dataset

train_path = './dataset/train/'
test_path = './dataset/test/'

# load train data
train_labels = dataset.load_image_labels(train_path)
train_images = dataset.load_image_data(train_path)

# load test data
test_labels = dataset.load_image_labels(test_path)
test_images = dataset.load_image_data(test_path)

# suffle train data
combined = list(zip(train_labels, train_images))
shuffle(combined)
train_labels, train_images = zip(*combined)
train_labels = np.array(train_labels)
train_images = np.array(train_images)

# suffle test data
combined = list(zip(test_labels, test_images))
shuffle(combined)
test_labels, test_images = zip(*combined)
test_labels = np.array(test_labels)
test_images = np.array(test_images)

# print(train_images.shape)
# print(test_images.shape)

# train_images = train_images / 255.0
# test_images = test_images / 255.0

# print(train_images.shape)
# print(test_images.shape)

# model layers creation
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(dataset.height, dataset.width)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# model compilation
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training
model.fit(train_images, train_labels, epochs=10)

# evaluation
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)