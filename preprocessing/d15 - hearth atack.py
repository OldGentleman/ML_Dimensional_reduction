import numpy as np
import pandas as pd
import zipfile
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

zf = zipfile.ZipFile('../data/raw_data/d15 - hearth atack.zip')
print(zipfile.ZipFile.namelist(zf)[0])
heart_attack_data = pd.read_csv(
    zf.open(zipfile.ZipFile.namelist(zf)[0]))


print(heart_attack_data)

heart_attack_data_target = heart_attack_data['DEATH_EVENT']
heart_attack_data_features = heart_attack_data.drop(
    'DEATH_EVENT', axis='columns')

max_abs_scaler = MaxAbsScaler()


for column in heart_attack_data_features.columns:
    if heart_attack_data_features[column].dtype != np.float64 or heart_attack_data_features[column].dtype != np.int64:
        heart_attack_data_features[column] = heart_attack_data_features[column].astype(
            'category')
        heart_attack_data_features[column] = heart_attack_data_features[column].cat.codes
        #heart_attack_data_features[column] = pd.DataFrame((ohe.fit_transform(heart_attack_data_features[column])))


heart_attack_data_train, heart_attack_data_test, heart_attack_data_train_targer, heart_attack_data_test_target=train_test_split(
    heart_attack_data_features, heart_attack_data_target, test_size=.2, random_state=71)


heart_attack_data_train_scaled=max_abs_scaler.fit_transform(
    heart_attack_data_train)
heart_attack_data_test_scaled=max_abs_scaler.fit_transform(
    heart_attack_data_test)

heart_attack_data_train_scaled=pd.DataFrame(
    heart_attack_data_train_scaled, columns=heart_attack_data_train.columns)
heart_attack_data_test_scaled=pd.DataFrame(
    heart_attack_data_test_scaled, columns=heart_attack_data_test.columns)


compression_train=dict(
    method='zip', archive_name='d15 - hearth atack_train.zip')

heart_attack_data_train_scaled.to_csv(
    '../data/preprocessed_data/d15 - hearth atack_train.zip', index=False, compression=compression_train)

compression_test=dict(method='zip', archive_name='d15 - hearth atack_test.zip')

heart_attack_data_test_scaled.to_csv(
    '../data/preprocessed_data/d15 - hearth atack_test.zip', index=False, compression=compression_test)

print(heart_attack_data)