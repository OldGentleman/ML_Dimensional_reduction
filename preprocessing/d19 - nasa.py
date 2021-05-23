import numpy as np
import pandas as pd
import zipfile
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

pd.set_option('display.max_columns', 500)

le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()
zf = zipfile.ZipFile('../data/raw_data/d19 - nasa.zip')
print(zipfile.ZipFile.namelist(zf)[-1])
nasa_data = pd.read_csv(zf.open(zipfile.ZipFile.namelist(zf)[-1]))
print(nasa_data)

nasa_data_target = nasa_data['Hazardous']
nasa_data_features = nasa_data.drop(['Hazardous'], 1)

# print(nasa_data_features)

for column in nasa_data_features.columns:
    if nasa_data_features[column].dtype != np.float64 or nasa_data_features[column].dtype != np.int64:
        nasa_data_features[column] = nasa_data_features[column].astype(
            'category')
        nasa_data_features[column] = nasa_data_features[column].cat.codes

#nasa_data_train, nasa_data_test, nasa_data_train_target, nasa_data_test_target = train_test_split(
    #nasa_data_features, nasa_data_target, test_size=.2, random_state=71)

max_abs_scaler = MaxAbsScaler()

nasa_data_X = max_abs_scaler.fit_transform(nasa_data_features)
nasa_data_Y= le.fit_transform(nasa_data_target)


print(nasa_data_Y)
# print(nasa_data_train_scaled)

nasa_data_X_scaled = pd.DataFrame(
    nasa_data_X, columns=nasa_data_features.columns)




nasa_data_X_scaled['target']=nasa_data_Y


# print(nasa_data_train_scaled)

compression = dict(method='zip', archive_name='d19 - nasa.csv')

nasa_data_X_scaled.to_csv(
    '../data/preprocessed_data/d19 - nasa.zip', index=False, compression=compression)
