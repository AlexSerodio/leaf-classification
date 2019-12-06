from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras    # 1.9.0 -> 1.15.0

import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

import prepare_dataset as dataset

root_path = './dataset-colorful/'
train_path = root_path + 'train/'
test_path = root_path + 'test/'

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

model = keras.Sequential([
    keras.layers.Conv2D(16, 5, padding='same', activation='relu', input_shape=(dataset.height, dataset.width, 3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 5, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 5, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    # keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# model compilation
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training
model.fit(train_images, train_labels, epochs=1)

# evaluation
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)