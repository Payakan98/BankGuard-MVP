"""
BankGuard — Fraud Detection Dashboard
Dark industrial UI with real-time alert monitoring.

Usage
─────
  python src/dashboard.py
  → http://localhost:5000
"""

import json
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

from flask import Flask, jsonify, render_template_string

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CFG

app = Flask(__name__)

# ── Data loader ───────────────────────────────────────────────────────────────

def load_alerts():
    """
    Supports two alert file formats:
      - New pipeline : alerts_{timestamp}.json  -> contains a JSON array
      - Old pipeline : alert_{uuid}.json        -> one JSON object per file
    """
    alerts_dir = Path(CFG.paths.alerts_dir)
    all_alerts = []

    # Format 1 — batched files (new pipeline)
    batched = sorted(alerts_dir.glob("alerts_*.json"), reverse=True)[:5]
    for f in batched:
        try:
            with open(f) as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    all_alerts.extend(data)
                else:
                    all_alerts.append(data)
        except Exception:
            pass

    # Format 2 — one file per alert (old pipeline), only if no batched files found
    if not all_alerts:
        single = sorted(alerts_dir.glob("alert_*.json"), reverse=True)[:200]
        for f in single:
            try:
                with open(f) as fh:
                    all_alerts.append(json.load(fh))
            except Exception:
                pass

    return [_normalize_alert(a) for a in all_alerts]


def _normalize_alert(a: dict) -> dict:
    """Map any legacy field names to the canonical schema the template expects."""
    return {
        "transaction_id":  a.get("transaction_id") or a.get("tx_id")       or a.get("id",           "N/A"),
        "card_id":         a.get("card_id")        or a.get("account_id")  or a.get("card",         "N/A"),
        "amount":          float(a.get("amount",    0)),
        "merchant_id":     a.get("merchant_id")    or a.get("merchant",    "N/A"),
        "timestamp":       a.get("timestamp")      or a.get("date",        ""),
        "anomaly_score":   float(a.get("anomaly_score") or a.get("score",  0.0)),
        "severity":        a.get("severity",       "MEDIUM"),
        "triggered_rules": a.get("triggered_rules") or a.get("rules")     or [],
        "detected_by":     a.get("detected_by")    or a.get("source",     "rules"),
        "generated_at":    a.get("generated_at",   ""),
    }


def compute_stats(alerts):
    if not alerts:
        return {"total": 0, "critical": 0, "high": 0, "medium": 0, "by_rules": 0}
    sev = [a.get("severity", "MEDIUM") for a in alerts]
    return {
        "total":    len(alerts),
        "critical": sev.count("CRITICAL"),
        "high":     sev.count("HIGH"),
        "medium":   sev.count("MEDIUM"),
        "by_rules": sum(1 for a in alerts if a.get("triggered_rules")),
    }


# ── HTML template ─────────────────────────────────────────────────────────────

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>BankGuard · Threat Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:        #080b0f;
    --surface:   #0d1117;
    --border:    #1c2330;
    --accent:    #00e5ff;
    --red:       #ff3b3b;
    --orange:    #ff8c00;
    --yellow:    #ffd600;
    --green:     #00e676;
    --muted:     #4a5568;
    --text:      #c9d1d9;
    --mono:      'Share Tech Mono', monospace;
    --sans:      'Barlow', sans-serif;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Scanline overlay */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,229,255,0.015) 2px,
      rgba(0,229,255,0.015) 4px
    );
    pointer-events: none;
    z-index: 100;
  }

  /* ── Header ── */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 32px;
    height: 60px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 50;
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 12px;
    font-family: var(--mono);
    font-size: 15px;
    letter-spacing: 0.12em;
    color: var(--accent);
    text-transform: uppercase;
  }

  .logo-icon {
    width: 28px; height: 28px;
    border: 2px solid var(--accent);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    box-shadow: 0 0 12px rgba(0,229,255,0.3);
    animation: pulse-border 3s ease-in-out infinite;
  }

  @keyframes pulse-border {
    0%, 100% { box-shadow: 0 0 8px rgba(0,229,255,0.3); }
    50%       { box-shadow: 0 0 20px rgba(0,229,255,0.6); }
  }

  .status-pill {
    display: flex; align-items: center; gap: 8px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--green);
    letter-spacing: 0.08em;
  }

  .status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    animation: blink 2s step-end infinite;
  }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.2; }
  }

  .header-time {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
  }

  /* ── Main layout ── */
  main { padding: 28px 32px; max-width: 1400px; margin: 0 auto; }

  /* ── Stat cards ── */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
    margin-bottom: 28px;
  }

  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    animation: fadein 0.4s ease both;
  }

  .stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }

  .stat-card.total::before   { background: var(--accent); }
  .stat-card.critical::before{ background: var(--red); }
  .stat-card.high::before    { background: var(--orange); }
  .stat-card.medium::before  { background: var(--yellow); }
  .stat-card.rules::before   { background: var(--green); }

  .stat-label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
  }

  .stat-value {
    font-family: var(--mono);
    font-size: 32px;
    font-weight: 700;
    line-height: 1;
  }

  .stat-card.total   .stat-value { color: var(--accent); }
  .stat-card.critical .stat-value{ color: var(--red); }
  .stat-card.high    .stat-value { color: var(--orange); }
  .stat-card.medium  .stat-value { color: var(--yellow); }
  .stat-card.rules   .stat-value { color: var(--green); }

  /* ── Table section ── */
  .table-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
  }

  .section-title {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
  }

  .refresh-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.1em;
    padding: 6px 14px;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .refresh-btn:hover {
    border-color: var(--accent);
    color: var(--accent);
  }

  .table-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }

  table { width: 100%; border-collapse: collapse; }

  thead tr {
    background: rgba(0,229,255,0.04);
    border-bottom: 1px solid var(--border);
  }

  th {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 12px 16px;
    text-align: left;
    white-space: nowrap;
  }

  tbody tr {
    border-bottom: 1px solid rgba(28,35,48,0.8);
    transition: background 0.15s;
    animation: fadein 0.3s ease both;
  }

  tbody tr:hover { background: rgba(0,229,255,0.03); }
  tbody tr:last-child { border-bottom: none; }

  td {
    padding: 11px 16px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text);
    white-space: nowrap;
  }

  .tx-id {
    color: var(--muted);
    font-size: 11px;
    max-width: 140px;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .card-id { color: var(--accent); }

  .amount { color: #e2e8f0; font-weight: 600; }

  .sev-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 10px;
    letter-spacing: 0.12em;
    padding: 3px 9px;
    border-radius: 3px;
    font-weight: 700;
  }

  .sev-CRITICAL { background: rgba(255,59,59,0.15);  color: var(--red);    border: 1px solid rgba(255,59,59,0.3); }
  .sev-HIGH     { background: rgba(255,140,0,0.15);  color: var(--orange); border: 1px solid rgba(255,140,0,0.3); }
  .sev-MEDIUM   { background: rgba(255,214,0,0.12);  color: var(--yellow); border: 1px solid rgba(255,214,0,0.25); }
  .sev-LOW      { background: rgba(74,85,104,0.2);   color: var(--muted);  border: 1px solid rgba(74,85,104,0.3); }

  .rule-tag {
    display: inline-block;
    background: rgba(0,229,255,0.08);
    border: 1px solid rgba(0,229,255,0.15);
    color: rgba(0,229,255,0.7);
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 3px;
    margin-right: 4px;
    letter-spacing: 0.06em;
  }

  .source-ml    { color: var(--green);  font-size: 11px; }
  .source-rules { color: var(--yellow); font-size: 11px; }
  .source-both  { color: var(--orange); font-size: 11px; }

  .empty-state {
    text-align: center;
    padding: 60px 0;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 12px;
    letter-spacing: 0.1em;
  }

  @keyframes fadein {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* ── Responsive ── */
  @media (max-width: 900px) {
    .stats-grid { grid-template-columns: repeat(3, 1fr); }
    main { padding: 20px 16px; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">⬡</div>
    BANKGUARD · THREAT MONITOR
  </div>
  <div class="status-pill">
    <span class="status-dot"></span>
    SYSTEM ACTIVE
  </div>
  <div class="header-time" id="clock">--:--:--</div>
</header>

<main>

  <!-- Stats row -->
  <div class="stats-grid">
    <div class="stat-card total">
      <div class="stat-label">Total Alerts</div>
      <div class="stat-value" id="s-total">{{ stats.total }}</div>
    </div>
    <div class="stat-card critical">
      <div class="stat-label">Critical</div>
      <div class="stat-value" id="s-critical">{{ stats.critical }}</div>
    </div>
    <div class="stat-card high">
      <div class="stat-label">High</div>
      <div class="stat-value" id="s-high">{{ stats.high }}</div>
    </div>
    <div class="stat-card medium">
      <div class="stat-label">Medium</div>
      <div class="stat-value" id="s-medium">{{ stats.medium }}</div>
    </div>
    <div class="stat-card rules">
      <div class="stat-label">Rule Hits</div>
      <div class="stat-value" id="s-rules">{{ stats.by_rules }}</div>
    </div>
  </div>

  <!-- Alerts table -->
  <div class="table-header">
    <span class="section-title">// Latest Alerts</span>
    <button class="refresh-btn" onclick="refresh()">↻ REFRESH</button>
  </div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Transaction ID</th>
          <th>Card / Account</th>
          <th>Amount</th>
          <th>Severity</th>
          <th>Score</th>
          <th>Rules Triggered</th>
          <th>Detected By</th>
          <th>Timestamp</th>
        </tr>
      </thead>
      <tbody id="alert-tbody">
        {% if alerts %}
          {% for a in alerts[:50] %}
          <tr>
            <td class="tx-id" title="{{ a.transaction_id }}">{{ a.transaction_id[:18] }}…</td>
            <td class="card-id">{{ a.card_id }}</td>
            <td class="amount">{{ "%.2f"|format(a.amount) }}</td>
            <td>
              <span class="sev-badge sev-{{ a.severity }}">
                {% if a.severity == 'CRITICAL' %}● {% elif a.severity == 'HIGH' %}◆ {% else %}◇ {% endif %}
                {{ a.severity }}
              </span>
            </td>
            <td style="color: var(--muted);">{{ "%.3f"|format(a.anomaly_score) }}</td>
            <td>
              {% for rule in a.triggered_rules %}
                <span class="rule-tag">{{ rule }}</span>
              {% endfor %}
            </td>
            <td>
              {% if a.detected_by == 'ensemble+rules' %}
                <span class="source-both">ensemble+rules</span>
              {% elif a.detected_by == 'ensemble' %}
                <span class="source-ml">ensemble</span>
              {% else %}
                <span class="source-rules">rules</span>
              {% endif %}
            </td>
            <td style="color: var(--muted); font-size:11px;">{{ a.timestamp[:19] }}</td>
          </tr>
          {% endfor %}
        {% else %}
          <tr><td colspan="8" class="empty-state">NO ALERTS DETECTED — SYSTEM NOMINAL</td></tr>
        {% endif %}
      </tbody>
    </table>
  </div>

</main>

<script>
  // Live clock
  function tick() {
    document.getElementById('clock').textContent =
      new Date().toLocaleTimeString('en-GB', {hour12: false});
  }
  tick(); setInterval(tick, 1000);

  // Refresh stats + table via API
  async function refresh() {
    try {
      const r = await fetch('/api/alerts');
      const data = await r.json();

      document.getElementById('s-total').textContent    = data.stats.total;
      document.getElementById('s-critical').textContent = data.stats.critical;
      document.getElementById('s-high').textContent     = data.stats.high;
      document.getElementById('s-medium').textContent   = data.stats.medium;
      document.getElementById('s-rules').textContent    = data.stats.by_rules;

      const tbody = document.getElementById('alert-tbody');
      if (!data.alerts.length) {
        tbody.innerHTML = '<tr><td colspan="8" class="empty-state">NO ALERTS DETECTED — SYSTEM NOMINAL</td></tr>';
        return;
      }

      const sevIcon = { CRITICAL: '●', HIGH: '◆', MEDIUM: '◇', LOW: '◇' };
      const srcClass = { 'ensemble+rules': 'source-both', ensemble: 'source-ml', rules: 'source-rules' };

      tbody.innerHTML = data.alerts.slice(0, 50).map(a => `
        <tr>
          <td class="tx-id" title="${a.transaction_id}">${a.transaction_id.slice(0,18)}…</td>
          <td class="card-id">${a.card_id}</td>
          <td class="amount">${parseFloat(a.amount).toFixed(2)}</td>
          <td><span class="sev-badge sev-${a.severity}">${sevIcon[a.severity]||'◇'} ${a.severity}</span></td>
          <td style="color:var(--muted)">${parseFloat(a.anomaly_score).toFixed(3)}</td>
          <td>${(a.triggered_rules||[]).map(r=>`<span class="rule-tag">${r}</span>`).join('')}</td>
          <td><span class="${srcClass[a.detected_by]||'source-ml'}">${a.detected_by}</span></td>
          <td style="color:var(--muted);font-size:11px">${String(a.timestamp).slice(0,19)}</td>
        </tr>
      `).join('');
    } catch(e) { console.error('Refresh failed', e); }
  }

  // Auto-refresh every 30s
  setInterval(refresh, 30000);
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    alerts = load_alerts()
    stats  = compute_stats(alerts)
    return render_template_string(TEMPLATE, alerts=alerts, stats=stats)


@app.route("/api/alerts")
def api_alerts():
    alerts = load_alerts()
    return jsonify({"alerts": alerts, "stats": compute_stats(alerts)})


@app.route("/api/score", methods=["POST"])
def api_score():
    """
    Score a single transaction in real time.
    
    POST /api/score
    {
        "transaction_id": "tx_001",
        "card_id": "acc_123",
        "amount": 2500.00,
        "merchant_id": "merch_456",
        "timestamp": "2026-03-14T22:00:00",
        "merchant_country": "CA",
        "channel": "online",
        "currency": "CAD",
        "ip_address": "192.168.1.1",
        "device_id": "dev_001",
        "status": "pending"
    }
    """
    import joblib
    from src.feature_engineering import build_features, get_feature_matrix
    from src.model_train import ensemble_score

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    try:
        df = pd.DataFrame([data])
        df = build_features(df, verbose=False)
        X  = get_feature_matrix(df)

        bundle    = joblib.load(CFG.paths.models_dir / "fraud_model.joblib")
        scores    = ensemble_score(X, bundle["if_pipe"], bundle["lof_pipe"])
        score     = float(scores[0])
        threshold = bundle["threshold"]

        def _severity(s):
            t = CFG.alerts.severity_thresholds
            if s >= t["CRITICAL"]: return "CRITICAL"
            if s >= t["HIGH"]:     return "HIGH"
            if s >= t["MEDIUM"]:   return "MEDIUM"
            return "LOW"

        return jsonify({
            "transaction_id": data.get("transaction_id", "N/A"),
            "anomaly_score":  round(score, 4),
            "threshold":      round(threshold, 4),
            "flagged":        score >= threshold,
            "severity":       _severity(score),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = CFG.dashboard
    print(f"\n  BankGuard Dashboard → http://{cfg.host}:{cfg.port}\n")
    app.run(host=cfg.host, port=cfg.port, debug=True)