# Autores: Alex Serodio Gonçalves e Luma Kühl

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

import dataset

root_path = './dataset-colored/'

# carrega labels e imagens
labels = dataset.load_image_labels(root_path)
images = dataset.load_image_data(root_path)

# randomiza o dataset carregado
combined = list(zip(labels, images))
shuffle(combined)
labels, images = zip(*combined)
labels = np.array(labels)
images = np.array(images)

# ponto de corte de 20% dos dados
test_size = int(20 / 100 * len(labels))

# corta 80% para treino
train_labels = labels[test_size:len(labels)-1]
train_images = images[test_size:len(labels)-1]

# corta 20% para teste
test_labels = labels[:test_size-1]
test_images = images[:test_size-1]

model = keras.Sequential([
    keras.layers.Conv2D(16, 5, padding='same', activation='relu', input_shape=(dataset.height, dataset.width, 3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 5, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 5, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# treino com dados de treino
model.fit(train_images, train_labels, epochs=10)

# validação com dados de teste
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)