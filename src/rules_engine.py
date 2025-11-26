from datetime import timedelta

def load_rules(path='rules/fraud_rules.yml'):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get('rules', [])

def evaluate(event, rules, state=None):
    matches = []
    amount = float(event.get('amount', 0))
    acc = event.get('account_id')
    for r in rules:
        cid = r.get('id')
        cond = r.get('condition', {})
        ok = True
        if 'amount_gt' in cond:
            ok = ok and (amount > cond['amount_gt'])
        if 'merchant_country_in' in cond:
            ok = ok and (event.get('merchant_country') in cond['merchant_country_in'])
        if 'tx_count_last_minutes' in cond and state is not None:
            m = cond['tx_count_last_minutes']['minutes']
            thr = cond['tx_count_last_minutes']['threshold']
            hist = state.get('tx_history', {}).get(acc, [])
            if hist:
                window_start = event.get('timestamp_parsed') if event.get('timestamp_parsed') else None
                # Si pas de timestamp_parsed, on skip la condition de vitesse
                if window_start:
                    cnt = sum(1 for t in hist if t >= window_start - timedelta(minutes=m))
                    ok = ok and (cnt >= thr)
                else:
                    ok = False
        if ok:
            matches.append({
                'id': cid,
                'severity': r.get('severity', 'low'),
                'description': r.get('description', '')
            })
    return matches
