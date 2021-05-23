import numpy as np
import pandas as pd
import zipfile
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
zf = zipfile.ZipFile('../data/raw_data/d10 - student_alchohol.zip')
drunk_students = pd.read_csv(zf.open(zipfile.ZipFile.namelist(zf)[1]))

target = drunk_students['g3']
drunk_students_features = drunk_students.drop(['g3'], 1)
# print(drunk_students_features.dtypes)

for column_name in drunk_students_features.columns:
    if(drunk_students_features[column_name].dtype != np.float64 and drunk_students_features[column_name].dtype != np.int64):
        drunk_students_features[column_name] = drunk_students_features[column_name].astype(
            'category')
        drunk_students_features[column_name] = drunk_students_features[column_name].cat.codes

drunk_students_train, drunk_students_test, target_train, target_test = train_test_split(
    drunk_students_features, target, test_size=.2, random_state=71)

max_abs_scaler = MaxAbsScaler()

drunk_students_train_scaled = max_abs_scaler.fit_transform(
    drunk_students_train)
drunk_students_test_scaled = max_abs_scaler.fit_transform(drunk_students_test)

drunk_students_train_scaled = pd.DataFrame(
    drunk_students_train_scaled, columns=drunk_students_train.columns)
drunk_students_test_scaled = pd.DataFrame(
    drunk_students_test_scaled, columns=drunk_students_test.columns)

compression_opts_train = dict(method='zip',
                              archive_name='d10 - student_alchohol_train.csv')
drunk_students_train_scaled.to_csv(
    '../data/preprocessed_data/d10 - student_alchohol_train.zip', index=False, compression=compression_opts_train)

compression_opts_test = dict(method='zip',
                             archive_name='d10 - student_alchohol_test.csv')
drunk_students_test_scaled.to_csv(
    '../data/preprocessed_data/d10 - student_alchohol_test.zip', index=False, compression=compression_opts_test)
