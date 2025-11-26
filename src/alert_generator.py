"""Simple aggregator that lists generated alerts and crée un rapport."""
import json
import glob

def summary(out='alerts/summary.json'):
    files = glob.glob('alerts/alert_*.json')
    alerts = []
    for f in files:
        with open(f) as fh:
            alerts.append(json.load(fh))
    with open(out, 'w') as of:
        json.dump({'n_alerts': len(alerts), 'alerts': alerts[:20]}, of, indent=2)
    print('Wrote', out)

if __name__ == '__main__':
    summary()
