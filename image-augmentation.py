# example of horizontal shift image augmentation
# adapted from 
# https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
import cv2

train_path = './dataset-colorful/train/'
output_path = './dataset-colorful/train/'
index = 1

def data_augmentation(train_path, output_path):
	global index
	
	image_count = 0
	labels = os.listdir(train_path)

	for label in labels:
		index = 1
		img = load_img(train_path + label)

		if label.endswith('.jpg'):
			label = label[:-4]

		data = img_to_array(img)				# convert to numpy array
		samples = expand_dims(data, 0)			# expand dimension to one sample

		datagen = ImageDataGenerator(width_shift_range=0.3)
		generate_images(samples, label, datagen)

		datagen = ImageDataGenerator(height_shift_range=0.3)
		generate_images(samples, label, datagen)

		datagen = ImageDataGenerator(horizontal_flip=True)
		generate_images(samples, label, datagen)

		datagen = ImageDataGenerator(rotation_range=90)
		generate_images(samples, label, datagen)

		datagen = ImageDataGenerator(zoom_range=[0.3,1.0])
		generate_images(samples, label, datagen)

		image_count += 1
		percentage = image_count / len(labels) * 100
		print('Processing: ', "{:10.1f}".format(percentage), '%')
	print('Data augmentation has finished!')

def generate_images(samples, label, datagen):
	global index
	it = datagen.flow(samples, batch_size=1)				# prepare iterator
	for i in range(9):
		pyplot.subplot(330 + 1 + i)							# define subplot
		batch = it.next()									# generate batch of images
		image = batch[0].astype('uint8')					# convert to unsigned integers for viewing
		cv2.imwrite(output_path + label + '-' + str(index) + '.jpg', image)
		index += 1

data_augmentation(train_path, output_path)