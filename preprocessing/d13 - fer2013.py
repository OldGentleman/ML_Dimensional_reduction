import numpy as np
import pandas as pd
import zipfile
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

zf = zipfile.ZipFile('../data/raw_data/d13 - fer2013.zip')
print(zipfile.ZipFile.namelist(zf)[1])
fer_data = pd.read_csv(
    zf.open(zipfile.ZipFile.namelist(zf)[1]))

ohe = preprocessing.OneHotEncoder(categories='auto')

fer_data = fer_data.drop('Usage', axis='columns')



fer_data_target = pd.get_dummies(fer_data.emotion)
fer_data_features = fer_data.drop(
    'emotion', axis='columns')


max_abs_scaler = MaxAbsScaler()

for column in fer_data_features.columns:
    if fer_data_features[column].dtype != np.float64 or fer_data_features[column].dtype != np.int64:
        fer_data_features[column] = fer_data_features[column].astype(
            'category')
        fer_data_features[column] = fer_data_features[column].cat.codes

fer_data_train, fer_data_test, fer_data_train_targer, fer_data_test_target=train_test_split(
    fer_data_features, fer_data_target, test_size=.2, random_state=71)


fer_data_train_scaled=max_abs_scaler.fit_transform(
    fer_data_train)
fer_data_test_scaled=max_abs_scaler.fit_transform(
    fer_data_test)

fer_data_train_scaled=pd.DataFrame(
    fer_data_train_scaled, columns=fer_data_train.columns)
fer_data_test_scaled=pd.DataFrame(
    fer_data_test_scaled, columns=fer_data_test.columns)


compression_train=dict(
    method='zip', archive_name='d13 - fer_train.zip')

fer_data_train_scaled.to_csv(
    '../data/preprocessed_data/d13 - fer_train.csv', index=False, compression=compression_train)

compression_test=dict(method='zip', archive_name='d13 - fer_test.zip')

fer_data_test_scaled.to_csv(
    '../data/preprocessed_data/d13 - fer_test.csv', index=False, compression=compression_test)


print(fer_data_target)
