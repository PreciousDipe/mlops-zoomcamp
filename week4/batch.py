#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd

year = int(sys.argv[2])  # 2023
month = int(sys.argv[3])  # 3
taxi_type = str(sys.argv[1])  # 'yellow'

url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/{year:04d}-{month:02d}.parquet'

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    
    return df

df = read_data(url)
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

y_pred.std()
print('predicted_mean_duration:',y_pred.mean())


df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred
df_result


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# The size of the output file is 66 MB
