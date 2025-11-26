import argparse
import os
import json
import pandas as pd
import numpy as np
import joblib

from src.preprocessing import featurize
from src.rules_engine import load_rules, evaluate

STATE = {'tx_history': {}}

def parse_timestamp(ts_val):
    try:
        # convertit en datetime, errors='coerce' → invalides → NaT
        ts = pd.to_datetime(ts_val, errors='coerce', utc=True)
        return ts.to_pydatetime() if not pd.isna(ts) else None
    except Exception:
        return None

def enrich(event):
    # pseudo géoloc pour la démo
    ip = event.get('ip_address', '')
    h = sum(ord(c) for c in ip)
    lat = (h % 180) - 90
    lon = (h % 360) - 180
    event['geo_lat'] = lat
    event['geo_lon'] = lon
    return event

def update_state(event, ts):
    acc = event.get('account_id')
    if not acc or ts is None:
        return
    STATE['tx_history'].setdefault(acc, []).append(ts)

def make_alert(event, reason):
    return {
        'tx_id': event.get('tx_id'),
        'account_id': event.get('account_id'),
        'amount': event.get('amount'),
        'timestamp': event.get('timestamp'),
        'reason': reason,
        'ml_score': event.get('ml_score'),
        'rules': event.get('rule_matches', [])
    }

def main(input_csv='data/transactions_sample.csv', model_path='models/fraud_model.joblib'):
    df = pd.read_csv(input_csv, dtype={'timestamp': str})  # forcer timestamp comme string
    df = featurize(df)
    bundle = joblib.load(model_path)
    model = bundle['model']
    scaler = bundle['scaler']
    rules = load_rules('rules/fraud_rules.yml')
    os.makedirs('alerts', exist_ok=True)

    for _, row in df.iterrows():
        event = row.to_dict()

        ts = parse_timestamp(event.get('timestamp'))
        if ts is None:
            # Optionnel : log ou print pour debug
            # print("Skipped event with invalid timestamp:", event)
            continue

        event = enrich(event)

        rule_matches = evaluate(event, rules, state=STATE)

        X = [[event.get('log_amount', 0), event.get('hour', 0)]]
        Xs = scaler.transform(X)
        pred = model.predict(Xs)
        score = float(model.decision_function(Xs)[0])

        event['ml_anomaly'] = int(pred[0] == -1)
        event['ml_score'] = score
        event['rule_matches'] = rule_matches

        alert = None
        if event['ml_anomaly'] == 1 or any(r.get('severity') in ('high','critical') for r in rule_matches):
            reason = []
            if event['ml_anomaly'] == 1:
                reason.append('ml_anomaly')
            reason += [r.get('id') for r in rule_matches]
            ts_str = ts.isoformat()  # ou str(ts)
            alert = {
                'tx_id': event.get('tx_id'),
                'account_id': event.get('account_id'),
                'amount': event.get('amount'),
                'timestamp': ts_str,
                'reason': reason,
                'ml_score': event.get('ml_score'),
                'rules': rule_matches
            }
            with open(os.path.join('alerts', f"alert_{event.get('tx_id')}.json"), 'w') as f:
                json.dump(alert, f, indent=2)
            print('ALERT:', alert['tx_id'], alert['reason'])

        update_state(event, ts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/transactions_sample.csv')
    parser.add_argument('--model', default='models/fraud_model.joblib')
    args = parser.parse_args()
    main(args.input, args.model)
