import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from flow import flow
from read.read_all_data import read_all_preprocessed_data, read_images


random_state = 228


for label, train, file_name in read_all_preprocessed_data():
    results = flow(train, label, dim_num=2, cv=10, random_state=random_state)
    results.to_csv(f'data/results/{file_name}_results.csv', index=False)

for label, train, file_name in read_images():
    results = flow(train, label, dim_num=2, cv=10, random_state=random_state)
    results.to_csv(f'data/results/{file_name}_results.csv', index=False)
