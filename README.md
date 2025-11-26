# BankGuard MVP — Fraud & Threat Detection System
BankGuard MVP est un prototype réaliste et pédagogique pour détecter des transactions frauduleuses et analyser des logs.

## Fonctionnalités (MVP)
- Génération d'un dataset synthétique de transactions.
- Moteur de règles déclaratives (YAML) pour détections basées sur seuils / blacklist.
- Détection d'anomalies (Isolation Forest) pour repérer les transactions suspectes.
- Pipeline simple ingestion → détection → alerting (format JSON).
- Notebook Jupyter d'analyse et visualisations.
- Dashboard Python minimal (Flask) pour visualiser les alertes .

## Structure du projet
bank-fraud-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│     └── generate_synthetic_transactions.py
├── data/transactions_sample.csv       
├── notebooks/
│     └── fraud_detection_analysis.ipynb
├── src/
│     ├── data_ingest.py
│     ├── preprocessing.py
│     ├── rules_engine.py
│     ├── model_train.py
│     ├── anomaly_detector.py
│     ├── alert_generator.py
│     └── dashboard.py
├── models/
│     └── fraud_model.joblib            
├── rules/
│     └── fraud_rules.yml
├── alerts/                             
└── images/                             

## Installation (local)
    python -m venv .venv
    source .venv/bin/activate   # Linux/macOS
    .venv\\Scripts\\activate     # Windows
    pip install -r requirements.txt

## Usage rapide (MVP)

1.Générer des données synthétiques :
    python data/generate_synthetic_transactions.py --out data/transactions_sample.csv --n 10000
2.Entraîner le modèle :
    python src/model_train.py --input data/transactions_sample.csv --out models/fraud_model.joblib
3.Lancer la détection / génération d’alertes :
    python clean_transactions.py
    python -m src.anomaly_detector --input data/transactions_sample.csv --model models/fraud_model.joblib

##Auteur: Islem CHOKRI
