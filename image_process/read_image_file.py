import numpy as np
import pandas as pd
import zipfile
from skimage.io import imread
from skimage.transform import resize
from typing import Tuple
from sklearn.preprocessing import LabelEncoder


def read_images(file_location: str, pattern: str, target_size: Tuple[int, int], num_of_records = -1):
    classes = []
    images = []
    with zipfile.ZipFile(file_location, 'r') as z:
        i = 1
        for file in z.namelist():
            if file.startswith(pattern):
                ifile = z.open(file)
                # optional set as_gray=True
                image = imread(ifile, as_gray=True)
                if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
                    image = resize(image, target_size, anti_aliasing=True)
                classes.append(ifile.name.split('/')[1])
                images.append(image.ravel())
                i += 1
                if i == num_of_records:
                    break

    image_matrix = np.stack(images)
    del images
    le = LabelEncoder()
    le.fit(classes)
    classes = le.transform(classes)
    return classes,  image_matrix


def read_flowers(file_location: str, target_size: Tuple[int, int], num_of_records = -1):
    classes = []
    images = []
    with zipfile.ZipFile(file_location, 'r') as z:
        i = 1
        for file in z.namelist():
            if file.endswith('.png'):
                ifile = z.open(file)
                image = imread(ifile, as_gray=True)
                if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
                    image = resize(image, target_size, anti_aliasing=True)
                classes.append(ifile.name.split('_')[0])
                images.append(image.ravel())
                i += 1
                if i == num_of_records:
                    break

    image_matrix = np.stack(images)
    del images
    le = LabelEncoder()
    le.fit(classes)
    classes = le.transform(classes)
    return classes,  image_matrix

