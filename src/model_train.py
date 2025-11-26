"""Train IsolationForest on basic features and save the model."""
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURES = ['log_amount', 'hour']

def featurize(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['log_amount'] = np.log1p(df['amount'])
    return df

def main(input_csv, out_model):
    df = pd.read_csv(input_csv, parse_dates=['timestamp'])
    df = featurize(df)
    X = df[FEATURES].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    clf.fit(Xs)
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    joblib.dump({'model': clf, 'scaler': scaler}, out_model)
    print('Saved model to', out_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input_csv', default='data/transactions_sample.csv')
    parser.add_argument('--out', dest='out_model', default='models/fraud_model.joblib')
    args = parser.parse_args()
    main(args.input_csv, args.out_model)
