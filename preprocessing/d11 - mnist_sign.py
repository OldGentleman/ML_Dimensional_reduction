import numpy as np
import pandas as pd
import zipfile
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

max_abs_scaler = MaxAbsScaler()

zf = zipfile.ZipFile('../data/raw_data/d11 - mnist_sign_language.zip')
print(zipfile.ZipFile.namelist(zf)[3])
sign_lang = pd.read_csv(
    zf.open(zipfile.ZipFile.namelist(zf)[3]))

    
print(sign_lang)

sign_lang_target = sign_lang['label']
sign_lang_features = sign_lang.drop('label', axis='columns')

sign_lang_scaled=max_abs_scaler.fit_transform(
    sign_lang_features)

sign_lang_scaled=pd.DataFrame(
    sign_lang_scaled, columns=sign_lang_features.columns)

sign_lang_scaled['target']=sign_lang_target

compression=dict(
    method='zip', archive_name='d11 - mnist_sign_language.csv')

sign_lang_scaled.to_csv(
    '../data/preprocessed_data/d11 - mnist_sign_language.zip', index=False, compression=compression)
