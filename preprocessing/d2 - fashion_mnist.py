import numpy as np
import pandas as pd
import zipfile
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
pd.set_option('display.max_columns', 500)
max_abs_scaler = MaxAbsScaler()

le = preprocessing.LabelEncoder()

zf = zipfile.ZipFile('../data/raw_data/d2 - fashion_mnist.zip')
print(zipfile.ZipFile.namelist(zf)[1])
data = pd.read_csv(
    zf.open(zipfile.ZipFile.namelist(zf)[1]))



print(data)

data_target = data['label']
data_features = data.drop('label', axis='columns')


for column in data_features.columns:
    if data_features[column].dtype != np.float64 or data_features[column].dtype != np.int64:
        data_features[column] = data_features[column].astype(
            'category')
        data_features[column] = le.fit_transform(data_features[column])
        #data_features[column] = data_features[column].cat.codes

data_features_scaled=max_abs_scaler.fit_transform(data_features)

print(data_features_scaled)


data_features_scaled=pd.DataFrame(data_features_scaled, columns=data_features.columns)

data_features_scaled['target']=data_target

compression=dict(
    method='zip', archive_name='d2 - fashion_mnist.csv')

data_features_scaled.to_csv(
    '../data/preprocessed_data/d2 - fashion_mnist.zip', index=False, compression=compression)