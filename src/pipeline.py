"""
BankGuard — Detection Pipeline Orchestrator
=============================================
Single entry-point that chains every stage:
  ingest → feature engineering → rules engine → ML scoring → alerting

Usage
─────
  python src/pipeline.py --input data/transactions_sample.csv

Options
  --model     path to model bundle     (default: models/fraud_model.joblib)
  --rules     path to rules YAML       (default: rules/fraud_rules.yml)
  --out-dir   where to write alerts    (default: alerts/)
  --dry-run   score without writing alerts to disk
  --log-level DEBUG|INFO|WARNING
"""

import argparse
import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CFG
from src.feature_engineering import build_features, get_feature_matrix
from src.model_train import ensemble_score


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO"):
    CFG.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            CFG.paths.logs_dir / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
        ),
    ]
    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt, handlers=handlers)


logger = logging.getLogger("bankguard.pipeline")


# ── Timing context manager ────────────────────────────────────────────────────

@contextmanager
def _timed(label: str):
    t0 = time.perf_counter()
    yield
    logger.info("  %-30s %.2fs", label, time.perf_counter() - t0)


# ── Alert data model ──────────────────────────────────────────────────────────

@dataclass
class Alert:
    transaction_id: str
    card_id: str
    amount: float
    merchant_id: str
    timestamp: str
    anomaly_score: float
    severity: str
    triggered_rules: list
    detected_by: str
    generated_at: str


def _severity(score: float) -> str:
    thresholds = CFG.alerts.severity_thresholds
    if score >= thresholds["CRITICAL"]:
        return "CRITICAL"
    if score >= thresholds["HIGH"]:
        return "HIGH"
    if score >= thresholds["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


# ── Stage 1 — Ingest ─────────────────────────────────────────────────────────

def stage_ingest(path) -> pd.DataFrame:
    logger.info("── Stage 1/5  INGEST ────────────────────────────")
    with _timed("read_csv"):
        df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info("  rows=%-8d  columns=%d", len(df), len(df.columns))

    # Normalise column names before validation (same map as feature_engineering)
    from src.feature_engineering import COLUMN_MAP
    df = df.rename(columns=COLUMN_MAP)

    # Basic schema validation (runs on normalised names)
    required = {"transaction_id", "card_id", "amount", "merchant_id", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    return df


# ── Stage 2 — Feature Engineering ────────────────────────────────────────────

def stage_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("── Stage 2/5  FEATURES ──────────────────────────")
    with _timed("build_features"):
        df = build_features(df, verbose=False)
    return df


# ── Stage 3 — Rules Engine ────────────────────────────────────────────────────

def stage_rules(df: pd.DataFrame, rules_path: Path) -> pd.DataFrame:
    logger.info("── Stage 3/5  RULES ─────────────────────────────")
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed — skipping rules engine")
        df["triggered_rules"] = [[] for _ in range(len(df))]
        return df

    if not rules_path.exists():
        logger.warning("Rules file not found: %s — skipping", rules_path)
        df["triggered_rules"] = [[] for _ in range(len(df))]
        return df

    with open(rules_path) as f:
        rules = yaml.safe_load(f)

    triggered = [[] for _ in range(len(df))]

    for rule in rules.get("rules", []):
        name = rule["name"]
        try:
            mask = df.eval(rule["condition"])
            for idx in df[mask].index:
                triggered[idx].append(name)
            logger.debug("  Rule %-35s matched %d rows", name, mask.sum())
        except Exception as exc:
            logger.warning("  Rule '%s' failed: %s", name, exc)

    df["triggered_rules"] = triggered
    total_flagged = sum(1 for t in triggered if t)
    logger.info("  Rules flagged %d / %d transactions", total_flagged, len(df))
    return df


# ── Stage 4 — ML Scoring ─────────────────────────────────────────────────────

def stage_ml_scoring(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    logger.info("── Stage 4/5  ML SCORING ────────────────────────")
    bundle = joblib.load(model_path)
    if_pipe   = bundle["if_pipe"]
    lof_pipe  = bundle["lof_pipe"]
    threshold = bundle["threshold"]

    X = get_feature_matrix(df)
    with _timed("ensemble_score"):
        scores = ensemble_score(X, if_pipe, lof_pipe)

    df["anomaly_score"] = scores
    df["ml_flagged"]    = scores >= threshold
    logger.info(
        "  threshold=%.3f  flagged=%d (%.2f%%)",
        threshold, df["ml_flagged"].sum(), 100 * df["ml_flagged"].mean()
    )
    return df, threshold


# ── Stage 5 — Alerting ────────────────────────────────────────────────────────

def stage_alerting(
    df: pd.DataFrame,
    threshold: float,
    out_dir: Path,
    dry_run: bool,
) -> list[Alert]:
    logger.info("── Stage 5/5  ALERTING ──────────────────────────")

    # Combine ML + rule triggers
    flagged = df[
        df["ml_flagged"] | df["triggered_rules"].apply(bool)
    ].copy()

    logger.info("  Combined flags: %d transactions", len(flagged))

    now = datetime.now(timezone.utc).isoformat()
    alerts = []
    for _, row in flagged.iterrows():
        score = float(row["anomaly_score"])
        by_ml    = score >= threshold
        by_rules = bool(row["triggered_rules"])
        detected_by = (
            "ensemble+rules" if (by_ml and by_rules)
            else "ensemble" if by_ml
            else "rules"
        )
        alerts.append(Alert(
            transaction_id  = str(row["transaction_id"]),
            card_id         = str(row["card_id"]),
            amount          = float(row["amount"]),
            merchant_id     = str(row["merchant_id"]),
            timestamp       = str(row["timestamp"]),
            anomaly_score   = round(score, 4),
            severity        = _severity(score),
            triggered_rules = row["triggered_rules"],
            detected_by     = detected_by,
            generated_at    = now,
        ))

    severity_counts = {}
    for a in alerts:
        severity_counts[a.severity] = severity_counts.get(a.severity, 0) + 1
    logger.info("  Severity breakdown: %s", severity_counts)

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"alerts_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(out_path, "w") as f:
            json.dump([asdict(a) for a in alerts], f, indent=2)
        logger.info("  Alerts written → %s", out_path)

    return alerts


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_pipeline(
    input_path: Path,
    model_path: Path,
    rules_path: Path,
    out_dir: Path,
    dry_run: bool = False,
) -> list[Alert]:
    t_start = time.perf_counter()
    logger.info("═" * 52)
    logger.info("BankGuard  Detection Pipeline  starting")
    logger.info("═" * 52)

    df                  = stage_ingest(input_path)
    df                  = stage_features(df)
    df                  = stage_rules(df, rules_path)
    df, threshold       = stage_ml_scoring(df, model_path)
    alerts              = stage_alerting(df, threshold, out_dir, dry_run)

    elapsed = time.perf_counter() - t_start
    logger.info("═" * 52)
    logger.info("Pipeline done in %.2fs  →  %d alerts", elapsed, len(alerts))
    logger.info("═" * 52)
    return alerts


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="BankGuard detection pipeline")
    p.add_argument("--input",     default=str(CFG.paths.data_raw))
    p.add_argument("--model",     default=str(CFG.paths.models_dir / "fraud_model.joblib"))
    p.add_argument("--rules",     default=str(CFG.paths.rules_file))
    p.add_argument("--out-dir",   default=str(CFG.paths.alerts_dir))
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    setup_logging(args.log_level)
    run_pipeline(
        input_path=Path(args.input),
        model_path=Path(args.model),
        rules_path=Path(args.rules),
        out_dir=Path(args.out_dir),
        dry_run=args.dry_run,
    )