import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from flow import flow
from read.read_all_data import read_all_preprocessed_data, read_images, read_iris_ibm_dataset, read_set


random_state = 228


for label, train, file_name in read_all_preprocessed_data():
    print(f'Started {file_name}')
    results = flow(train, label, dim_num=2, cv=5, random_state=random_state)
    results.to_csv(f'static_tests/{file_name}_results.csv', index=False)
    print(f'Finished {file_name}')

# for label, train, file_name in read_images():
#     # print(f'Started {file_name}')
#     print(file_name)
#     print(train.shape)
#     # results = flow(train, label, dim_num=2, cv=5, random_state=random_state)
#     # results.to_csv(f'data/results/{file_name}_results.csv', index=False)
#     # print(f'Finished {file_name}')

# for label, train, file_name in [read_set('d22 - student_exams.zip')]:
#     results = flow(train, label, dim_num=2, cv=10, random_state=random_state)
#     results.to_csv(f'data/{file_name}.csv', index=False)