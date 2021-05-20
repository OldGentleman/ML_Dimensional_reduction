import numpy as np
import pandas as pd
import zipfile
from skimage.io import imread
from skimage.transform import resize
from typing import Tuple
from sklearn.preprocessing import LabelEncoder


def take_hand_gestures_image_class_name(line: str) -> str:
    return line.split('/')[0]


def take_food_image_class_name(line: str) -> str:
    return line.split('/')[1]


def take_flower_image_class_name(line: str) -> str:
    return line.split('/')[-1].split('_')[0]


sets_name = {'flower': take_flower_image_class_name,
                   'food': take_food_image_class_name, 'hand_gestures': take_hand_gestures_image_class_name}


def read_images(file_location: str, file_format: str, image_size: Tuple[int, int], set_name: str, num_of_records=-1):
    classes = []
    images = []
    take_class_name = sets_name[set_name]
    with zipfile.ZipFile(file_location, 'r') as z:
        i = 1
        for file in z.namelist():
            if file.endswith(file_format):
                ifile = z.open(file)
                image = imread(ifile, as_gray=True)
                if image.shape != image_size:
                    image = resize(image, image_size, anti_aliasing=True)
                classes.append(take_class_name(ifile.name))
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
