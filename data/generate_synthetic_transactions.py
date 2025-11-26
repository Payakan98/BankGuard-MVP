"""Generate synthetic transactions CSV with injected fraud patterns."""
import argparse
import random
import csv
import uuid
import ipaddress
from datetime import datetime, timezone, timedelta

COUNTRY_LIST = ['CA','US','GB','CN','RU','NG','BR','FR','DE','IN']
MERCHANTS = [f"merchant_{i}" for i in range(1,201)]

def rand_ip():
    return str(ipaddress.IPv4Address(random.getrandbits(32)))

def generate_row(base_time, account_id, fraud=False):
    ts = base_time + timedelta(seconds=random.randint(0,300))
    if fraud:
        amount = round(random.uniform(5000,20000), 2)
        merchant = random.choice(MERCHANTS[:20])
        country = random.choice(['RU','NG','CN'])
        status = random.choice(['success','failed'])
    else:
        amount = round(random.expovariate(1/80) + 1, 2)
        merchant = random.choice(MERCHANTS)
        country = random.choice(COUNTRY_LIST)
        status = 'success'
    return {
        'tx_id': str(uuid.uuid4()),
        'timestamp': ts.isoformat(),
        'account_id': f'acc_{account_id}',
        'amount': amount,
        'currency': 'CAD',
        'merchant_id': merchant,
        'merchant_country': country,
        'channel': random.choice(['web','mobile','branch']),
        'ip_address': rand_ip(),
        'device_id': f'dev_{random.randint(1,5000)}',
        'status': status,
        'label': 'fraud' if fraud else 'genuine'
    }

def main(out='data/transactions_sample.csv', n=10000, fraud_rate=0.01):
    with open(out, 'w', newline='') as csvfile:
        fieldnames = [
            'tx_id','timestamp','account_id','amount','currency',
            'merchant_id','merchant_country','channel',
            'ip_address','device_id','status','label'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        base_time = datetime.now(timezone.utc)
        for i in range(n):
            account_id = random.randint(1,2000)
            is_fraud = random.random() < fraud_rate
            row = generate_row(base_time + timedelta(seconds=i*2), account_id, fraud=is_fraud)
            writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/transactions_sample.csv')
    parser.add_argument('--n', type=int, default=10000)
    parser.add_argument('--fraud_rate', type=float, default=0.01)
    args = parser.parse_args()
    main(out=args.out, n=args.n, fraud_rate=args.fraud_rate)
