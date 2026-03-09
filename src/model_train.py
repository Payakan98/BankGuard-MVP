"""
BankGuard — Model Training & Evaluation
========================================
Trains an ensemble of unsupervised anomaly detectors:
  • Isolation Forest  (primary)
  • Local Outlier Factor (secondary)

When ground-truth labels are available the script evaluates the ensemble
with AUPRC, F1, Precision and Recall and prints a full classification report.

Usage
─────
  python src/model_train.py \\
      --input  data/transactions_sample.csv \\
      --output models/

Optional flags
  --eval        compute metrics against is_fraud label (if present)
  --shap        generate SHAP feature importance plot
  --no-cache    retrain even if a saved model already exists
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CFG
from src.feature_engineering import FEATURE_COLUMNS, build_features, get_feature_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Model wrappers ────────────────────────────────────────────────────────────

def _build_isolation_forest() -> Pipeline:
    """IsolationForest wrapped in a RobustScaler pipeline."""
    return Pipeline([
        ("scaler", RobustScaler()),
        ("clf", IsolationForest(
            n_estimators=CFG.model.if_n_estimators,
            max_samples=CFG.model.if_max_samples,
            contamination=CFG.model.if_contamination,
            random_state=CFG.model.if_random_state,
            n_jobs=-1,
        )),
    ])


def _build_lof() -> Pipeline:
    """LocalOutlierFactor wrapped in a RobustScaler pipeline (novelty=True for predict)."""
    return Pipeline([
        ("scaler", RobustScaler()),
        ("clf", LocalOutlierFactor(
            n_neighbors=CFG.model.lof_n_neighbors,
            contamination=CFG.model.lof_contamination,
            novelty=True,      # enables .predict() / .score_samples() at inference time
            n_jobs=-1,
        )),
    ])


# ── Score normalisation ───────────────────────────────────────────────────────

def _normalise(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; higher = more anomalous."""
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def ensemble_score(X: np.ndarray, if_pipe: Pipeline, lof_pipe: Pipeline) -> np.ndarray:
    """
    Weighted average of normalised anomaly scores from both models.
    score_samples() returns negative outlier scores → negate so higher = worse.
    """
    w_if, w_lof = CFG.model.ensemble_weights

    if_raw  = -if_pipe.named_steps["clf"].score_samples(
        if_pipe.named_steps["scaler"].transform(X)
    )
    lof_raw = -lof_pipe.named_steps["clf"].score_samples(
        lof_pipe.named_steps["scaler"].transform(X)
    )

    return w_if * _normalise(if_raw) + w_lof * _normalise(lof_raw)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)

    auprc   = average_precision_score(labels, scores)
    auc_roc = roc_auc_score(labels, scores)
    prec, rec, _ = precision_recall_curve(labels, scores)

    report = classification_report(labels, preds, target_names=["Legit", "Fraud"], output_dict=True)

    metrics = {
        "auprc":        round(auprc,   4),
        "auc_roc":      round(auc_roc, 4),
        "threshold":    threshold,
        "fraud_f1":     round(report["Fraud"]["f1-score"],   4),
        "fraud_prec":   round(report["Fraud"]["precision"],  4),
        "fraud_recall": round(report["Fraud"]["recall"],     4),
        "support_fraud": int(labels.sum()),
    }

    logger.info("── Evaluation metrics ───────────────────────────")
    for k, v in metrics.items():
        logger.info("  %-20s %s", k, v)

    # print readable report
    print("\n" + classification_report(labels, preds, target_names=["Legit", "Fraud"]))
    return metrics


def _optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return threshold that maximises F1 on the fraud class."""
    prec, rec, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1[:-1])   # last element has no threshold
    return float(thresholds[best_idx])


# ── SHAP explainability ───────────────────────────────────────────────────────

def _shap_importance(if_pipe: Pipeline, X: np.ndarray, output_dir: Path):
    try:
        import shap
        import matplotlib.pyplot as plt

        X_scaled = if_pipe.named_steps["scaler"].transform(X)
        explainer = shap.TreeExplainer(if_pipe.named_steps["clf"])
        shap_values = explainer.shap_values(X_scaled[:2000])   # sample for speed

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_scaled[:2000], feature_names=FEATURE_COLUMNS,
                          show=False, plot_size=None)
        plt.tight_layout()
        out = output_dir / "shap_summary.png"
        plt.savefig(out, dpi=150)
        logger.info("SHAP summary saved → %s", out)
    except ImportError:
        logger.warning("shap not installed — skipping SHAP plot. pip install shap")
    except Exception as exc:
        logger.warning("SHAP failed: %s", exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def train(input_path: Path, output_dir: Path, run_eval: bool, run_shap: bool, no_cache: bool):
    model_path    = output_dir / "fraud_model.joblib"
    metrics_path  = output_dir / "metrics.json"

    if model_path.exists() and not no_cache:
        logger.info("Cached model found at %s  (use --no-cache to retrain)", model_path)
        return

    # ── Load & engineer features ──────────────────────────────────────────────
    logger.info("Loading data from %s", input_path)
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    logger.info("Rows: %d   Columns: %s", len(df), list(df.columns))

    df = build_features(df)
    X  = get_feature_matrix(df)
    logger.info("Feature matrix  shape=%s", X.shape)

    # ── Train models ──────────────────────────────────────────────────────────
    logger.info("Training Isolation Forest …")
    t0 = time.perf_counter()
    if_pipe = _build_isolation_forest()
    if_pipe.fit(X)
    logger.info("  IF done in %.1fs", time.perf_counter() - t0)

    logger.info("Training Local Outlier Factor …")
    t0 = time.perf_counter()
    lof_pipe = _build_lof()
    lof_pipe.fit(X)
    logger.info("  LOF done in %.1fs", time.perf_counter() - t0)

    # ── Ensemble scores ───────────────────────────────────────────────────────
    scores = ensemble_score(X, if_pipe, lof_pipe)
    df["anomaly_score"] = scores

    # ── Evaluation (optional) ─────────────────────────────────────────────────
    metrics = {}
    threshold = CFG.model.alert_threshold

    if run_eval and "is_fraud" in df.columns:
        labels = df["is_fraud"].to_numpy()
        threshold = _optimal_threshold(scores, labels)
        logger.info("Optimal threshold (max-F1): %.4f", threshold)
        metrics = evaluate(scores, labels, threshold)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved → %s", metrics_path)

    # ── Save artefacts ────────────────────────────────────────────────────────
    bundle = {
        "if_pipe":    if_pipe,
        "lof_pipe":   lof_pipe,
        "threshold":  threshold,
        "features":   FEATURE_COLUMNS,
        "metrics":    metrics,
    }
    joblib.dump(bundle, model_path, compress=3)
    logger.info("Model bundle saved → %s", model_path)

    # ── SHAP ─────────────────────────────────────────────────────────────────
    if run_shap:
        _shap_importance(if_pipe, X, output_dir)

    logger.info("✓ Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="BankGuard model training")
    p.add_argument("--input",   default=str(CFG.paths.data_raw),    help="CSV input path")
    p.add_argument("--output",  default=str(CFG.paths.models_dir),  help="Output directory")
    p.add_argument("--eval",    action="store_true", help="Evaluate with is_fraud label")
    p.add_argument("--shap",    action="store_true", help="Generate SHAP plot")
    p.add_argument("--no-cache",action="store_true", help="Force retrain")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        run_eval=args.eval,
        run_shap=args.shap,
        no_cache=args.no_cache,
    )
