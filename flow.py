import numpy as np
import pandas as pd
import time

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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import zipfile
from skimage.io import imread
from skimage.transform import resize
from typing import Tuple

from image_process.read_image_file import read_images, read_flowers


# train and targets - apart
def flow(train_data, targets, dim_num=2, cv=5, only_one=False) -> pd.DataFrame:

    results = pd.DataFrame(columns=['Reduction_method', 'Classificator', 'Unreduced_acc',
                                    'Reduced_acc', 'Clf_unreducted_time', 'Clf_reducted_time', 'Reduction_time'])
    reduction_names = ['UMAP', 'PCA', 'KPCA', 'LDA']
    reductions = [
        umap.UMAP(n_components=dim_num),
        PCA(n_components=dim_num),
        KernelPCA(n_components=dim_num, kernel='rbf'),
        LinearDiscriminantAnalysis(n_components=dim_num)
    ]

    classifieres_names = ['RBF SVM', 'Nearest Neighbors',
                          'Naive Bayes', 'Random Forest', 'Neural Network']
    classifieres = [
        SVC(kernel='rbf', gamma='auto', class_weight='balanced', C=1e3),
        KNeighborsClassifier(3),
        GaussianNB(),
        RandomForestClassifier(n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000)]

    i = 0
    for r_name, r_method in zip(reduction_names, reductions):

        reduction_start_time = time.time()
        if r_name == 'LDA':
            reduced_train = r_method.fit(
                train_data, targets).transform(train_data)
        else:
            reduced_train = r_method.fit_transform(train_data)
        reduction_finish_time = time.time()

        max_abs = MaxAbsScaler().fit(reduced_train)
        reduced_train = max_abs.transform(reduced_train)

        for clf_name, clf in zip(classifieres_names, classifieres):
            # print(f'{r_name} {clf_name}')

            unscored_start_time = time.time()
            unreduced_score = cross_val_score(
                clf, train_data, targets, cv=cv).mean()
            unscored_finish_time = time.time()
            # print(f'unreduced score: {unreduced_score}')

            scored_start_time = time.time()
            reduced_score = cross_val_score(
                clf, reduced_train, targets, cv=cv).mean()
            scored_finish_time = time.time()
            # print(f'reduced score: {reduced_score}')

            results.loc[i] = [r_name, clf_name, unreduced_score, reduced_score,
                              unscored_finish_time-unscored_start_time, scored_finish_time-scored_start_time, reduction_finish_time-reduction_start_time]
            i += 1

            if only_one:
                break

        if only_one:
            break
    return results.sort_values(by=['Classificator', 'Reduction_method'])
# print('read food take a lot of memory')
# target_food, images_food = read_images(
#     'data/raw_data/d5 - food_rec_I.zip', 'images/', (512, 512))
# flow_images(images_food, target_food)


print('read flowers')
target_flowers, images_target = read_flowers(
    'data/raw_data/d4 - flower_rec_i.zip', (128, 128), 140)
results_flowers = flow(images_target, target_flowers, dim_num=2, cv=5)
print(results_flowers)

# print('wine')
# wine = pd.read_csv('data/preprocessed_data/wine.zip')
# target = wine['target']
# wine_train = wine.drop(columns=['target'])
# results = flow(wine_train, target, 2, 5)
# print(results)


print('iris')
data = pd.read_csv('data/preprocessed_data/iris.zip')
target = data['target']
data_train = data.drop(columns=['target'])
results = flow(data_train, target, 2, 5)
print(results)
