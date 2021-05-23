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
#ohe = preprocessing.OneHotEncoder()
zf = zipfile.ZipFile('../data/raw_data/d10 - student_alchohol.zip')
print(zipfile.ZipFile.namelist(zf)[1])
data = pd.read_csv(
    zf.open(zipfile.ZipFile.namelist(zf)[1]))


columns_to_encode = ['school','famsize','mjob','fjob','reason','guardian']
colums_to_le = ['sex','pstatus','schoolsup','famsup','paid','activities','nursery','higher','internet']
data = data.drop(['address','romantic','g1','g2'], axis='columns')

#print(data)

data_target = data['g3']
data_features = data.drop('g3', axis='columns')


'''for column in data_features.columns:
    if data_features[column].dtype != np.float64 or data_features[column].dtype != np.int64:
        print(data_features[column])
        data_features[column] = data_features[column].astype(
            'category')
        data_features[column] = le.fit_transform(data_features[column])
        print(data_features[column])
        data_features[column] = ohe.fit_transform(data_features[column])
        print(data_features[column])'''

for column in colums_to_le:
    data_features[column] = le.fit_transform(data_features[column])

for column_name in columns_to_encode:
    data_features[column_name] = data_features[column_name].astype('category')
    data_features[column_name] = data_features[column_name].cat.codes

encoder = ce.OneHotEncoder(cols=columns_to_encode, return_df=True)

data_features_encoded = encoder.fit_transform(data_features)

print(data_features_encoded)
print('---------------------------------------------------------------------')

data_features_scaled=max_abs_scaler.fit_transform(data_features)




data_features_scaled=pd.DataFrame(data_features_scaled, columns=data_features.columns)

data_features_scaled['target']=data_target

compression=dict(
    method='zip', archive_name='d10 - student_alchohol.csv')

data_features_scaled.to_csv(
    '../data/preprocessed_data/d10 - student_alchohol.zip', index=False, compression=compression)