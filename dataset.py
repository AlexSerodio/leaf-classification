# Autores: Alex Serodio Gonçalves e Luma Kühl

import os
import cv2
import numpy as np

file_extension = '.jpg'

width = 100
height = 150

def resize(path, output):
    file_names = os.listdir(path)
    for file in file_names:
        image = cv2.imread(path + '/' + file)
        image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
        cv2.imwrite(output + file, image)

def resize_and_grayscale(path, output):
    file_names = os.listdir(path)
    for file in file_names:
        image = cv2.imread(path + '/' + file)
        image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
        cv2.imwrite(output + file, image)

def resize_and_threshold(path, output):
    file_names = os.listdir(path)
    for file in file_names:
        image = cv2.imread(path + '/' + file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
        (thresh, image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite(output + file, image)

def load_image_labels(path):
    file_names = os.listdir(path)
    labels = map_label(file_names)
    return labels

def load_image_data(path):
    images = np.array([])
    images = []
    file_names = os.listdir(path)
    for file in file_names:
        image = cv2.imread(path + '/' + file)
        images.append(image)

    return images

def map_label(file_names):
    for i in range(len(file_names)):
        name = format_name(file_names[i])
        file_names[i] = get_name(name)

    return file_names

def get_name(name):
    if 1060 <= name <= 1122:
        return 0 # 'Chinese horse'
    elif 1552 <= name <= 1616:
        return 1 # 'Anhui Barberry'
    elif 1123 <= name <= 1194:
        return 2 # 'Chinese redbud'
    elif 1195 <= name <= 1267:
        return 3 # 'True indigo'
    elif 2051 <= name <= 2113:
        return 4 # 'Japanese cheesewood'
    elif 2166 <= name <= 2230:
        return 5 # 'Camphortree'
    elif 2347 <= name <= 2423:
        return 6 # 'Deodar'
    elif 2547 <= name <= 2612:
        return 7 # 'Oleander'
    elif 3111 <= name <= 3175:
        return 8 # 'Chinese Toon'
    elif 3447 <= name <= 3510:
        return 9 # 'Canadian popular'

def format_name(name):
    if name.endswith(file_extension):
        name = name[:len(file_extension)]
    return int(name.split('-')[0])

# resize_and_threshold('./dataset-colored', './dataset-threshold/')