import numpy as np
import pandas as pd
import glob
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

def read_data(dataroot, file_ending='*.pcap_ISCX.csv'):
    filenames = glob.glob(f"{dataroot}/{file_ending}")
    if not filenames:
        raise ValueError(f"No files found in {dataroot} with pattern {file_ending}")
    combined_csv = dd.read_csv(filenames, dtype=object)
    combined_csv.columns = combined_csv.columns.str.strip()
    return combined_csv

def preprocess_data(data):
    print('stripped column names')
    data = data.rename(columns=lambda x: x.strip())
    print('dropped bad columns')
    data = data.drop(columns=['Flow Packets/s', 'Flow Bytes/s', 'Label'])
    
    nan_count = data.isnull().sum().sum()
    print('There are {} nan entries'.format(nan_count))
    
    if nan_count > 0:
        data = data.apply(lambda x: x.fillna(x.mean()), axis=0)
        print('filled NAN')

    data = data.astype(float).apply(pd.to_numeric)
    print('converted to numeric')
    
    if data.isnull().sum().sum() != 0:
        raise ValueError("There should not be any NaN")

    return data

def load_data(dataroot):
    with ProgressBar():
        data = read_data(dataroot,'*.pcap_ISCX.csv').compute()
    num_records, num_features = data.shape
    print(f"there are {num_records} flow records with {num_features} feature dimension")
    df_label = data['Label']
    data = preprocess_data(data)
    X = normalize(data.values)
    y = encode_label(df_label.values)
    return (X, y)

# Remaining code is unchanged
