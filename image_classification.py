import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

import umap
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

import zipfile
from skimage.io import imread
from skimage.transform import resize
from typing import Tuple

from image_process.read_image_file import read_images, read_flowers


def flow_images(train_data, targets):

    reduction_names = ['UMAP', 'PCA', 'KPCA', 'LDA']
    dim_num = 2
    reductions = [
        umap.UMAP(n_components=dim_num),
        PCA(n_components=dim_num),
        KernelPCA(n_components=dim_num, kernel='rbf'),
        LinearDiscriminantAnalysis(n_components=dim_num)
    ]

    classifieres_names = ['RBF SVM', 'Nearest Neighbors', 'Naive Bayes']
    classifieres = [
        SVC(gamma='scale', C=1),
        KNeighborsClassifier(3),
        GaussianNB()]

    for r_name, r_method in zip(reduction_names, reductions):
        train_food, test_food, train_target, test_target = train_test_split(
            train_data, targets, test_size=0.2, random_state=228)

        if r_name == 'LDA':
            reduced_train = r_method.fit(
                train_food, train_target).transform(train_food)
        else:
            reduced_train = r_method.fit_transform(train_food)
        reduced_test = r_method.transform(test_food)

        max_abs = MaxAbsScaler().fit(reduced_train)
        reduced_train = max_abs.transform(reduced_train)
        reduced_test = max_abs.transform(reduced_test)

        for clf_name, clf in zip(classifieres_names, classifieres):
            print(f'{r_name} {clf_name}')

            clf.fit(train_food, train_target)
            print(f'unreduced score: {clf.score(test_food, test_target)}')

            clf.fit(reduced_train, train_target)
            print(f'reduced score: {clf.score(reduced_test, test_target)}')


# target_food, images_food = read_images(
#     'd5-food_rec_I.zip', 'images/', (512, 512), 10)

# print('Encoding')
# le = LabelEncoder()
# le.fit(target_food)
# target_food = le.transform(target_food)


# flow_images(images_food, target_food)

target_flowers, images_target = read_flowers(
    'data/raw_data/d4-flower_rec_i.zip', 'images/', (128, 128))

print('Encoding')
le = LabelEncoder()
le.fit(target_flowers)
target_food = le.transform(target_food)


flow_images(images_target, target_food)