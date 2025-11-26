"""Preprocessing helpers."""

import pandas as pd
import numpy as np

def featurize(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['log_amount'] = np.log1p(df['amount'])
    return df
