"""
BankGuard — Centralized Configuration
All tuneable parameters live here. No magic numbers in source files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ── Project root (two levels up from this file) ───────────────────────────────
ROOT = Path(__file__).resolve().parent


# ── Path layout ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Paths:
    data_raw: Path = ROOT / "data" / "transactions_sample.csv"
    data_processed: Path = ROOT / "data" / "transactions_processed.csv"
    models_dir: Path = ROOT / "models"
    alerts_dir: Path = ROOT / "alerts"
    rules_file: Path = ROOT / "rules" / "fraud_rules.yml"
    logs_dir: Path = ROOT / "logs"

    def __post_init__(self):
        for p in (self.models_dir, self.alerts_dir, self.logs_dir):
            p.mkdir(parents=True, exist_ok=True)


# ── Data generation ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DataConfig:
    n_transactions: int = 50_000
    fraud_rate: float = 0.015          # ~1.5 % — realistic imbalance
    random_seed: int = 42
    currencies: List[str] = field(default_factory=lambda: ["EUR", "USD", "GBP", "CHF"])
    high_risk_countries: List[str] = field(
        default_factory=lambda: ["NG", "RO", "UA", "VN", "PK"]
    )
    high_risk_mcc_codes: List[int] = field(
        default_factory=lambda: [7995, 6051, 5912, 4829]   # gambling, crypto, pharma, wire
    )


# ── Feature engineering ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class FeatureConfig:
    velocity_windows: List[int] = field(default_factory=lambda: [1, 6, 24])  # hours
    amount_log_transform: bool = True
    encode_hour_cyclically: bool = True        # sin/cos encoding of hour-of-day
    merchant_risk_min_tx: int = 10             # min tx to compute merchant risk score


# ── Model ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelConfig:
    # Isolation Forest
    if_contamination: float = 0.015
    if_n_estimators: int = 200
    if_max_samples: str = "auto"
    if_random_state: int = 42

    # Local Outlier Factor (comparison)
    lof_n_neighbors: int = 20
    lof_contamination: float = 0.015

    # Ensemble: weights [IsolationForest, LOF]
    ensemble_weights: List[float] = field(default_factory=lambda: [0.65, 0.35])

    # Scoring threshold to emit an alert (0–1)
    alert_threshold: float = 0.60

    # Cross-validation folds for evaluation
    cv_folds: int = 5


# ── Alerting ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AlertConfig:
    max_alerts_per_run: int = 5_000
    output_format: str = "json"          # "json" | "csv"
    severity_thresholds: dict = field(
        default_factory=lambda: {
            "CRITICAL": 0.85,
            "HIGH":     0.70,
            "MEDIUM":   0.60,
        }
    )


# ── Dashboard ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    reload_interval_ms: int = 30_000     # polling interval for live data


# ── Master config (single import point) ───────────────────────────────────────
@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


CFG = Config()   # singleton — import this everywhere