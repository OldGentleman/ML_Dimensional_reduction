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

from util.read_image_file import read_images_to_df

target_food, images_food = read_images_to_df('data/raw_data/food.zip', 'images/', (512, 512))

food_image_umap = umap.UMAP(n_components = 2, n_neighbors = 4)
reduced_foods_images = food_image_umap.fit_transform(images_food)

print(reduced_foods_images)