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
train_data = dataset.load_image_data(train_path)

# load test data
test_labels = dataset.load_image_labels(test_path)
test_data = dataset.load_image_data(test_path)

# suffle train data
combined = list(zip(train_labels, train_data))
shuffle(combined)
train_labels, train_data = zip(*combined)

# suffle test data
combined = list(zip(test_labels, test_data))
shuffle(combined)
test_labels, test_data = zip(*combined)

print(len(train_labels))
print(len(train_data))
print(train_data[0].shape)

print('+------------------+')

print(len(test_labels))
print(len(test_data))
print(test_data[0].shape)