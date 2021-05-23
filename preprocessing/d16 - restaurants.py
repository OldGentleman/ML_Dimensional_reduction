import numpy as np
import pandas as pd
import zipfile
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()
zf = zipfile.ZipFile('../data/raw_data/d16 - restaurants.zip')
print(zipfile.ZipFile.namelist(zf)[-1])
restaurant_data = pd.read_csv(
    zf.open(zipfile.ZipFile.namelist(zf)[-1]), encoding="ISO-8859-1")

restaurant_data = restaurant_data.drop(
    'Rating color', axis='columns').drop('Rating text', axis='columns')
print(restaurant_data)

restaurant_data_target = restaurant_data['Aggregate rating']
restaurant_data_features = restaurant_data.drop(
    'Aggregate rating', axis='columns')

max_abs_scaler = MaxAbsScaler()


for column in restaurant_data_features.columns:
    if restaurant_data_features[column].dtype != np.float64 or restaurant_data_features[column].dtype != np.int64:
        restaurant_data_features[column] = restaurant_data_features[column].astype(
            'category')
        restaurant_data_features[column] = restaurant_data_features[column].cat.codes
        #restaurant_data_features[column] = pd.DataFrame((ohe.fit_transform(restaurant_data_features[column])))


restaurant_data_train, restaurant_data_test, restaurant_data_train_targer, restaurant_data_test_target=train_test_split(
    restaurant_data_features, restaurant_data_target, test_size=.2, random_state=71)


restaurant_data_train_scaled=max_abs_scaler.fit_transform(
    restaurant_data_train)
restaurant_data_test_scaled=max_abs_scaler.fit_transform(
    restaurant_data_test)

#restaurant_data_train_targer = pd.get_dummies(restaurant_data_train_targer)

print((restaurant_data_target))
#print((restaurant_data_train_scaled))

#restaurant_data_train_scaled=np.concatenate(restaurant_data_train_scaled,restaurant_data_train_targer)
#restaurant_data_train_targer = le.fit_transform(restaurant_data_train_targer)
#restaurant_data_test_target = le.fit_transform(restaurant_data_test_target)

restaurant_data_train_scaled['target']=restaurant_data_target


#print(restaurant_data_train_targer)

restaurant_data_train_scaled=pd.DataFrame(
    restaurant_data_train_scaled, columns=restaurant_data_train.columns)
restaurant_data_test_scaled=pd.DataFrame(
    restaurant_data_test_scaled, columns=restaurant_data_test.columns)


compression_train=dict(
    method='zip', archive_name='d16 - restaurant_train.zip')

restaurant_data_train_scaled.to_csv(
    '../data/preprocessed_data/d16 - restaurant_train.zip', index=False, compression=compression_train)

compression_test=dict(method='zip', archive_name='d16 - restaurant_test.zip')

restaurant_data_test_scaled.to_csv(
    '../data/preprocessed_data/d16 - restaurant_test.zip', index=False, compression=compression_test)

#print(restaurant_data)
