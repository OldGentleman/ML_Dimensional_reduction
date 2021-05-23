import numpy as np
import pandas as pd
import zipfile
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

pd.set_option('display.max_columns', 500)

le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()
zf = zipfile.ZipFile('../data/raw_data/d16 - restaurants.zip')
print(zipfile.ZipFile.namelist(zf)[-1])
restaurant_data = pd.read_csv(
    zf.open(zipfile.ZipFile.namelist(zf)[-1]), encoding="ISO-8859-1")

restaurant_data = restaurant_data.drop(columns=['Restaurant ID', 'Restaurant Name', 'City', 'Address', 'Locality', 'Locality Verbose', 'Cuisines', 'Currency', 'Aggregate rating', 'Rating color'], axis=1)
print(restaurant_data)

restaurant_data_target = restaurant_data['Rating text']
restaurant_data_features = restaurant_data.drop(
    'Rating text', axis='columns')

max_abs_scaler = MaxAbsScaler()


for column in restaurant_data_features.columns:
    if restaurant_data_features[column].dtype != np.float64 or restaurant_data_features[column].dtype != np.int64:
        restaurant_data_features[column] = restaurant_data_features[column].astype(
            'category')
        restaurant_data_features[column] = restaurant_data_features[column].cat.codes
        #restaurant_data_features[column] = pd.DataFrame((ohe.fit_transform(restaurant_data_features[column])))



restaurant_data_scaled=max_abs_scaler.fit_transform(restaurant_data_features)

restaurant_data_target = le.fit_transform(restaurant_data_target)


print(restaurant_data_target)


restaurant_data_scaled=pd.DataFrame(
    restaurant_data_scaled, columns=restaurant_data_features.columns)

restaurant_data_scaled['target']=restaurant_data_target

compression=dict(
    method='zip', archive_name='d16 - restaurant.csv')

restaurant_data_scaled.to_csv(
    '../data/preprocessed_data/d16 - restaurant.zip', index=False, compression=compression)

#print(restaurant_data)
