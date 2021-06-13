import numpy as np
import pandas as pd
import zipfile
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
zf = zipfile.ZipFile('../data/raw_data/d13 - fer2013_I.zip')
print(zipfile.ZipFile.namelist(zf)[1])
fer_data = pd.read_csv(
    zf.open(zipfile.ZipFile.namelist(zf)[1]))

ohe = preprocessing.OneHotEncoder(categories='auto')

fer_data = fer_data.drop('Usage', axis='columns')



fer_data_target = fer_data['emotion']
fer_data_features = fer_data.drop(
    'emotion', axis='columns')


max_abs_scaler = MaxAbsScaler()

for column in fer_data_features.columns:
    if fer_data_features[column].dtype != np.float64 or fer_data_features[column].dtype != np.int64:
        fer_data_features[column] = fer_data_features[column].astype(
            'category')
        fer_data_features[column] = fer_data_features[column].cat.codes


fer_data_scaled=max_abs_scaler.fit_transform(
    fer_data_features)


fer_data_scaled=pd.DataFrame(
    fer_data_scaled, columns=fer_data_features.columns)

fer_data_Y = le.fit_transform(fer_data_target)


fer_data_scaled['target'] = fer_data_Y

compression=dict(
    method='zip', archive_name='d13 - fer.csv')

fer_data_scaled.to_csv(
    '../data/preprocessed_data/d13 - fer.zip', index=False, compression=compression)



print(fer_data_target)
