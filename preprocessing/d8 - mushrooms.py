import numpy as np
import pandas as pd
import zipfile
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import category_encoders as ce


pd.set_option('display.max_columns', 500)
max_abs_scaler = MaxAbsScaler()

le = preprocessing.LabelEncoder()

zf = zipfile.ZipFile('../data/raw_data/d8 - mushrooms.zip')
print(zipfile.ZipFile.namelist(zf)[0])
data = pd.read_csv(
    zf.open(zipfile.ZipFile.namelist(zf)[0]))


print(data)

data_target = data['cap-color']
data_features = data.drop('cap-color', axis='columns')

for column_name in data_features.columns:
    data_features[column_name] = data_features[column_name].astype('category')
    data_features[column_name] = data_features[column_name].cat.codes


encoder = ce.OneHotEncoder(cols=data_features.columns, return_df=True)

data_features_encoded = encoder.fit_transform(data_features)

target_encoded = le.fit_transform(data_target)


#data_features_scaled=pd.DataFrame(data_features_scaled, columns=data_features.columns)

data_features_encoded['target']=target_encoded

print(data_features_encoded)

compression=dict(
    method='zip', archive_name='d8 - mushrooms.csv')

data_features_encoded.to_csv(
    '../data/preprocessed_data/d8 - mushrooms.zip', index=False, compression=compression)