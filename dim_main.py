from read.read_all_data import read_all_preprocessed_data, read_images, read_iris_ibm_dataset, read_set
from flow import flow, flow_for_pair
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import warnings
from sklearn.svm import SVC
import umap
warnings.filterwarnings("ignore")


random_state = 228
# sets = [(*read_set('iris.zip'), [1, 2, 3]), (*read_set('d16 - restaurant.zip'), [1, 3, 5, 7, 9, 11]),
#      (*read_set('d8 - mushrooms.zip'), [1, 3, 5, 7, 9, 11]), (*read_set('d22 - student_exams.zip'), [1, 3, 5, 7, 9, 11])]
sets = [(*read_set('iris.zip'), [1, 2, 3]),
        (*read_set('d2 - fashion_mnist.zip'), [2, 5, 10, 20, 50]),
        (*read_set('d8 - mushrooms.zip'), [2, 4, 5, 7, 8, 10]),
        (*read_set('d9 - mnist_sighns.zip'), [2, 5, 10, 20, 50]),
        (*read_set('d10 - student_alchohol.zip'), [2, 4, 5, 7, 8, 10]),
        (*read_set('d11 - mnist_sign_language.zip'), [2, 5, 10, 20, 50]),
        (*read_set('d13 - fer2013.zip'), [2, 5, 10, 20, 50]),
        (*read_set('d16 - restaurant.zip'), [2, 4, 5, 7, 8, 10]),
        (*read_set('d22 - student_exams.zip'), [2, 4, 5, 7, 8, 10]),
        (*read_set('IBM.zip'), [2, 4, 5, 7, 8, 10]),
        (*read_set('wine.zip'), [2, 4, 5, 7, 8, 10]),
        (*read_set('d12 - human.zip'), [2, 4, 5, 7, 8, 10])]
# sets = [(*read_set('d11 - mnist_sign_language.zip'), [2, 4, 5, 7, 8, 10])]
# print(read_set('d13 - fer2013_I.zip'))
for label, train, file_name, dims in sets:
    scores = []
    times = []
    print(file_name)
    for dim in dims:
        clf = SVC(kernel='rbf', gamma='auto', class_weight='balanced', C=1e3)
        r_method = umap.UMAP(n_components=dim, random_state=random_state)
        score, time = flow_for_pair(
            train, label, r_method, clf, dim_num=dim, cv=2, random_state=random_state)
        scores.append(score)
        times.append(time)
    print(np.mean(scores), scores)
    print(np.mean(times), times)
    print('--------------')

# for label, train, file_name in read_images():
#     scores = []
#     times = []
#     print(file_name)
#     dims = [2, 50, 100, 500, 2000, 5000, 10000, 15000]
#     for dim in dims:
#         clf = SVC(kernel='rbf', gamma='auto', class_weight='balanced', C=1e3)
#         r_method = umap.UMAP(n_components=dim, random_state=random_state)
#         score, time = flow_for_pair(
#             train, label, r_method, clf, dim_num=dim, cv=2, random_state=random_state)
#         scores.append(score)
#         times.append(time)
#     print(np.mean(scores), scores)
#     print(np.mean(times), times)
#     print('--------------')
