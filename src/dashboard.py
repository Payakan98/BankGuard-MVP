from flask import Flask, render_template_string, jsonify
import glob
import json
import os

app = Flask(__name__)

TEMPLATE = '''
<html><head><title>BankGuard MVP — Dashboard</title></head><body>
<h2>BankGuard MVP — Alerts</h2>
<p>Generated alerts (latest 50)</p>
<ul>
{% for a in alerts %}
  <li><b>{{a['tx_id']}}</b> – {{a['account_id']}} – {{a['amount']}} – {{a['reason']}}</li>
{% endfor %}
</ul>
</body></html>
'''

def load_alerts(limit=50):
    files = sorted(glob.glob('alerts/alert_*.json'), reverse=True)
    alerts = []
    for f in files[:limit]:
        with open(f) as fh:
            alerts.append(json.load(fh))
    return alerts

@app.route('/alerts.json')
def alerts_json():
    return jsonify(load_alerts())

@app.route('/')
def index():
    return render_template_string(TEMPLATE, alerts=load_alerts())

if __name__ == '__main__':
    os.makedirs('alerts', exist_ok=True)
    app.run(debug=True, port=5000)
