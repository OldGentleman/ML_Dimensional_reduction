import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

import umap
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA, KernelPCA

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

from util.read_image_file import read_images_to_df

target_food, images_food = read_images_to_df(
    'data/raw_data/food.zip', 'images/', (512, 512))

print('Encoding')
le = LabelEncoder()
le.fit(target_food)
target_food = le.transform(target_food)

train_food, test_food, train_target, test_target = train_test_split(
    images_food, target_food, test_size=0.2, random_state=228)

print('UMAP')
food_image_umap = umap.UMAP(n_components=2, n_neighbors=4)
reduced_train = food_image_umap.fit_transform(train_food)
reduced_test = food_image_umap.transform(test_food)


print('max abs encoding')

max_abs = MaxAbsScaler().fit(reduced_train)
reduced_train = max_abs.transform(reduced_train)
reduced_test = max_abs.transform(reduced_test)

print('SVM')
svc_noramal = SVC(gamma='scale', C=1)
svc_reduced = SVC(gamma='scale', C=1)

svc_noramal.fit(train_food, train_target)
svc_reduced.fit(reduced_train, train_target)

print(svc_noramal.score(test_food, test_target))
print(svc_reduced.score(reduced_test, test_target))
