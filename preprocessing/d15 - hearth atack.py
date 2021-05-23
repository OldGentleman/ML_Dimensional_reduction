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

heart_attack_data_scaled=max_abs_scaler.fit_transform(
    heart_attack_data_features)


heart_attack_data_scaled=pd.DataFrame(
    heart_attack_data_scaled, columns=heart_attack_data_features.columns)


heart_attack_data_scaled['target'] = heart_attack_data_target

compression=dict(
    method='zip', archive_name='d15 - hearth atack.csv')

heart_attack_data_scaled.to_csv(
    '../data/preprocessed_data/d15 - hearth atack.zip', index=False, compression=compression)



print(heart_attack_data)