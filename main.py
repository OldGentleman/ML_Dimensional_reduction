import pandas as pd
import numpy as np

from flow import flow


# print('read food take a lot of memory')
# target_food, images_food = read_images(
#     'data/raw_data/d5 - food_rec_I.zip', 'images/', (512, 512))
# flow_images(images_food, target_food)


# print('flowers')
# labels_flower, images_flower = rif.read_images(
#     file_location='data/raw_data/d4 - flower_rec_i.zip', file_format='.png',
#     image_size=(128, 128), set_name='flower',
#     num_of_records=140)
# results_flower = flow(train_data=images_flower,
#                       targets=labels_flower, dim_num=2, cv=5)
# print(results_flower)

# print('hand gestures')
# labels_hand_gestures, images_hand_gestures = rif.read_images(
#     file_location='data/raw_data/d20 - hand_gestures_I.zip', file_format='.jpg', image_size=(320, 320), set_name='hand_gestures')
# results_hand_gestures = flow(
#     train_data=images_hand_gestures, targets=labels_hand_gestures, dim_num=2, cv=5)
# print(results_hand_gestures)

# print('wine')
# wine = pd.read_csv('data/preprocessed_data/wine.zip')
# labels_wine = wine['target']
# train_wine = wine.drop(columns=['target'])
# results = flow(train_wine, labels_wine, 2, 5)
# print(results)

print('iris')
iris = pd.read_csv('data/preprocessed_data/iris.zip')
labels_iris = iris['target']
train_iris = iris.drop(columns=['target'])
results = flow(train_iris, labels_iris, 2, 5)
results.to_csv('data/results/iris_results.csv', index=False)
print(results)
