#!/usr/bin/env python
# coding: utf-8

import click
import pickle
import numpy as np
import pandas as pd

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

@click.command()
@click.option(
    "--year",
    default=2022,
    help="Year of NYC taxi trip data"
)
@click.option(
    "--month",
    default=2,
    help="Month of NYC taxi trip data"
)
def score(year: int, month: int):
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # ### Q1: Standard deviation of predicted duration
    # import numpy as np
    # np.std(y_pred)
    # ### Q5: Mean of predicted duration
    print(f'Mean of predicted duration {year:04d}-{month:02d}:', np.mean(y_pred))

    # ### Q2: Preparing output parquet with pyarrow
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = pd.DataFrame({'ride_id': df['ride_id'], 'predicted_duration': y_pred})

    output_file = 'data/{year:04d}-{month:02d}-predictions.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

if __name__ == '__main__':
    score()
