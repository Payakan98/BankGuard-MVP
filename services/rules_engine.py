"""Apply declarative rules from YAML against an event (row dict).
This is a minimal evaluator for the provided fraud_rules.yml.
"""
import yaml
import math
from datetime import datetime, timedelta


def load_rules(path='rules/fraud_rules.yml'):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg['rules']


def evaluate(event, rules, state=None):
    """event: dict with fields (amount, timestamp, merchant_country, ...)
    state: optional dict to compute temporal features (tx history per account)
    Returns list of matched rule ids.
    """
    matches = []
    amount = float(event.get('amount',0))
    ts = datetime.fromisoformat(event.get('timestamp'))
    acc = event.get('account_id')

    for r in rules:
        cid = r['id']
        cond = r.get('condition',{})
        ok = True
        if 'amount_gt' in cond:
            ok = ok and (amount > cond['amount_gt'])
        if 'merchant_country_in' in cond:
            ok = ok and (event.get('merchant_country') in cond['merchant_country_in'])
        if 'tx_count_last_minutes' in cond and state is not None:
            m = cond['tx_count_last_minutes']['minutes']
            thr = cond['tx_count_last_minutes']['threshold']
            # count tx in state within window
            hist = state.get(acc, [])
            window_start = ts - timedelta(minutes=m)
            cnt = sum(1 for t in hist if t >= window_start)
            ok = ok and (cnt >= thr)
        if 'distance_km_gt' in cond and state is not None:
            # state should contain last_login_location per account as (lat,lon,timestamp)
            last = state.get('last_login', {}).get(acc)
            if last:
                last_ts = last['ts']
                if (ts - last_ts).total_seconds() <= cond.get('last_login_within_hours', 0)*3600:
                    # compute euclidean approx (not great but ok for MVP)
                    dx = float(event.get('geo_lat',0)) - last['lat']
                    dy = float(event.get('geo_lon',0)) - last['lon']
                    dist_km = math.sqrt(dx*dx + dy*dy) * 111  # rough conversion
                    ok = ok and (dist_km > cond['distance_km_gt'])
                else:
                    ok = False
        if ok:
            matches.append({'id': cid, 'severity': r.get('severity','low'), 'description': r.get('description','')})
    return matches
