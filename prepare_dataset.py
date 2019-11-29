import os
from PIL import Image 
import glob
import matplotlib.image as img
import numpy as np

file_extension = '.jpg'

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def label_translator(file_names):
    for i in range(len(file_names)):
        name = format_name(file_names[i])

        if 1060 <= name <= 1122:
            name = 'Chinese horse'
        elif 1552 <= name <= 1616:
            name = 'Anhui Barberry'
        elif 1123 <= name <= 1194:
            name = 'Chinese redbud'
        elif 1195 <= name <= 1267:
            name = 'True indigo'
        elif 2051 <= name <= 2113:
            name = 'Japanese cheesewood'
        elif 2166 <= name <= 2230:
            name = 'Camphortree'
        elif 2347 <= name <= 2423:
            name = 'Deodar'
        elif 2547 <= name <= 2612:
            name = 'Oleander'
        elif 3111 <= name <= 3175:
            name = 'Chinese Toon'
        elif 3447 <= name <= 3510:
            name = 'Canadian popular'

        file_names[i] = name

    return file_names

def format_name(name):
    if name.endswith(file_extension):
        name = name[:len(file_extension)]
    return int(name)

def load_image_labels(path):
    file_names = os.listdir(path)
    labels = label_translator(file_names)
    return labels

def load_image_data(path):
    images = []
    file_names = os.listdir(path)
    for file in file_names:
        image = img.imread(path+'/'+file)
        image = rgb2gray(image)
        image = np.resize(image, (150, 200))
        images.append(image)
    return images;

labels = load_image_labels('./dataset-slim')
data = load_image_data('./dataset-slim')

print(labels[0])
print(data[0])
