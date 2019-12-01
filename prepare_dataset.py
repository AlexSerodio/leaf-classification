import os
import cv2
import numpy as np

file_extension = '.jpg'

def prepare_image_data(path, width, height, output):
    file_names = os.listdir(path)
    for file in file_names:
        image = cv2.imread(path + '/' + file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
        cv2.imwrite(output + file, image)

def load_image_labels(path):
    file_names = os.listdir(path)
    labels = map_label(file_names)
    return labels

def load_image_data(path):
    images = []
    file_names = os.listdir(path)
    for file in file_names:
        image = cv2.imread(path + '/' + file, cv2.IMREAD_GRAYSCALE)
        images.append(np.asarray(image))
    return images

def map_label(file_names):
    for i in range(len(file_names)):
        name = format_name(file_names[i])
        file_names[i] = get_name(name)

    return file_names

def get_name(name):
    if 1060 <= name <= 1122:
        return 'Chinese horse'
    elif 1552 <= name <= 1616:
        return 'Anhui Barberry'
    elif 1123 <= name <= 1194:
        return 'Chinese redbud'
    elif 1195 <= name <= 1267:
        return 'True indigo'
    elif 2051 <= name <= 2113:
        return 'Japanese cheesewood'
    elif 2166 <= name <= 2230:
        return 'Camphortree'
    elif 2347 <= name <= 2423:
        return 'Deodar'
    elif 2547 <= name <= 2612:
        return 'Oleander'
    elif 3111 <= name <= 3175:
        return 'Chinese Toon'
    elif 3447 <= name <= 3510:
        return 'Canadian popular'


def format_name(name):
    if name.endswith(file_extension):
        name = name[:len(file_extension)]
    return int(name)