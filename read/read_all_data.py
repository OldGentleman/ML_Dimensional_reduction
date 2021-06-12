from os import listdir
import pandas as pd
import read.read_image_file as rif

image_path = 'data/raw_data'

image_params = [[f'{image_path}/d21 - KFC,McDonald_rec_I.zip',
                 'jpg', (128, 128), 'd21', -1],
                [f'{image_path}/d20 - hand_gestures_I.zip',
                 'jpg', (128, 128), 'd20', -1],
                [f'{image_path}/d18 - blood cells_I.zip',
                 'jpeg', (128, 128), 'd18', -1],
                [f'{image_path}/d17 - gemstones_I.zip',
                 'jpg', (128, 128), 'd17', -1],
                [f'{image_path}/d5 - food_rec_I.zip',
                 'jpg', (128, 128), 'd5', -1],
                [f'{image_path}/d4 - flower_rec_i.zip',
                 'png', (128, 128), 'd4', -1],
                [f'{image_path}/d3 - flower_rec_I.zip',
                 'jpg', (128, 128), 'd3', -1],
                [f'{image_path}/d1 - frut_rec_I.zip',
                 'jpg', (128, 128), 'd1', -1]
                ]


def read_images(number_of_records=None):

    data_images = []

    for param in image_params:
        data_images.append(rif.read_images(
            file_location=param[0], file_format=param[1], image_size=param[2], set_name=param[3], num_of_records=param[4]) + tuple([param[3]]))

    return data_images


def read_all_preprocessed_data():
    preprocessed_data = []
    path = 'data/preprocessed_data'
    for file_name in listdir(path):
        if file_name.split('.')[0][-1].lower() != 'i':
            data = pd.read_csv(f'{path}/{file_name}')  
            labels = data['target']
            train_data = data.drop(columns=['target'])
            train_data = train_data.to_numpy()

            preprocessed_data.append((labels, train_data, file_name.split('.')[0]))

    return preprocessed_data

def read_iris_ibm_dataset():
    preprocessed_data = []
    path = 'data/preprocessed_data'
    for file_name in listdir(path):
        if file_name == 'iris.zip' or file_name == 'IBM.zip':
            data = pd.read_csv(f'{path}/{file_name}')
            labels = data['target']
            train_data = data.drop(columns=['target'])
            train_data = train_data.to_numpy()

            preprocessed_data.append((labels, train_data, file_name.split('.')[0]))

    return preprocessed_data

def read_set(set_name):
    preprocessed_data = []
    path = 'data/preprocessed_data'
    for file_name in listdir(path):
        if file_name == set_name:
            data = pd.read_csv(f'{path}/{file_name}')
            labels = data['target']
            train_data = data.drop(columns=['target'])
            train_data = train_data.to_numpy()

            return (labels, train_data, file_name.split('.')[0])