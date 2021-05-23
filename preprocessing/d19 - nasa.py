import numpy as np
import pandas as pd
import zipfile
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()
zf = zipfile.ZipFile('../data/raw_data/d19 - nasa.zip')
print(zipfile.ZipFile.namelist(zf)[-1])
nasa_data = pd.read_csv(zf.open(zipfile.ZipFile.namelist(zf)[-1]))


nasa_data_target = nasa_data['Hazardous']
nasa_data_features = nasa_data.drop(['Hazardous'], 1)

# print(nasa_data_features)

for column in nasa_data_features.columns:
    if nasa_data_features[column].dtype != np.float64 or nasa_data_features[column].dtype != np.int64:
        nasa_data_features[column] = nasa_data_features[column].astype(
            'category')
        nasa_data_features[column] = nasa_data_features[column].cat.codes

nasa_data_train, nasa_data_test, nasa_data_train_target, nasa_data_test_target = train_test_split(
    nasa_data_features, nasa_data_target, test_size=.2, random_state=71)

max_abs_scaler = MaxAbsScaler()

nasa_data_train_scaled = max_abs_scaler.fit_transform(nasa_data_train)
nasa_data_test_scaled = max_abs_scaler.fit_transform(nasa_data_test)
#nasa_data_train_target = ohe.fit_transform(nasa_data_train_target.values.reshape(-1,1))
nasa_data_train_target = le.fit_transform(nasa_data_train_target)
nasa_data_test_target = le.fit_transform(nasa_data_test_target)

print(nasa_data_train_target)
# print(nasa_data_train_scaled)

nasa_data_train_scaled = pd.DataFrame(
    nasa_data_train_scaled, columns=nasa_data_train.columns)
nasa_data_test_scaled = pd.DataFrame(
    nasa_data_test_scaled, columns=nasa_data_test.columns)



nasa_data_train_scaled['target']=nasa_data_train_target
nasa_data_test_scaled['target']=nasa_data_test_target

# print(nasa_data_train_scaled)

compression_train = dict(method='zip', archive_name='d19 - nasa_train.zip')

nasa_data_train_scaled.to_csv(
    '../data/preprocessed_data/d19 - nasa_train.zip', index=False, compression=compression_train)

compression_test = dict(method='zip', archive_name='d19 - nasa_test.zip')

nasa_data_test_scaled.to_csv(
    '../data/preprocessed_data/d19 - nasa_test.zip', index=False, compression=compression_test)
