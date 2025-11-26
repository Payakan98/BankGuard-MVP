"""Simple data ingestion helper (CSV reader)."""
import pandas as pd

def read_csv(path='data/transactions_sample.csv', n=None):
    if n:
        return pd.read_csv(path, parse_dates=['timestamp'], nrows=n)
    return pd.read_csv(path, parse_dates=['timestamp'])

if __name__ == '__main__':
    df = read_csv()
    print('Loaded', len(df), 'rows')
