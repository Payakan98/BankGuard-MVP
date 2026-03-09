"""
BankGuard — Feature Engineering
Transforms raw transactions into a rich feature matrix.

Features produced
─────────────────
• amount_log           log1p-normalised amount
• hour_sin / hour_cos  cyclic encoding of hour-of-day
• is_weekend           binary weekend flag
• country_risk         0/1 high-risk country flag (optional col)
• mcc_risk             0/1 high-risk MCC code flag (optional col)
• velocity_{w}h_count  number of tx by same card in last w hours
• velocity_{w}h_sum    total spend by same card in last w hours
• amount_vs_mean_ratio tx amount vs card historical mean
• merchant_risk_score  fraud-rate proxy per merchant (smoothed)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import CFG

logger = logging.getLogger(__name__)


# ── Column name normalisation ─────────────────────────────────────────────────
# Maps your CSV column names → internal names used throughout the pipeline.
# Add / change entries here whenever the source schema changes.

COLUMN_MAP = {
    "tx_id":            "transaction_id",
    "account_id":       "card_id",
    "merchant_country": "country",
    "label":            "is_fraud",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cyclic_encode(series: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    """Encode a cyclic variable (e.g. hour 0-23) as (sin, cos) pair."""
    angle = 2 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if CFG.features.encode_hour_cyclically:
        df["hour_sin"], df["hour_cos"] = _cyclic_encode(df["hour"], 24)
    else:
        df["hour_norm"] = df["hour"] / 23.0

    return df


def _add_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Country risk — column may be absent in some datasets
    if "country" in df.columns:
        df["country_risk"] = df["country"].isin(CFG.data.high_risk_countries).astype(int)
    else:
        logger.warning("Column 'country' not found — country_risk set to 0")
        df["country_risk"] = 0

    # MCC risk — column is optional
    if "mcc_code" in df.columns:
        df["mcc_risk"] = df["mcc_code"].isin(CFG.data.high_risk_mcc_codes).astype(int)
    else:
        logger.warning("Column 'mcc_code' not found — mcc_risk set to 0")
        df["mcc_risk"] = 0

    return df


def _add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling transaction count and spend per card over multiple time windows.
    Requires a sorted DataFrame with a datetime 'timestamp' column.
    """
    df = df.sort_values("timestamp").copy()
    df = df.set_index("timestamp")

    for window_h in CFG.features.velocity_windows:
        window_str = f"{window_h}h"
        count_col  = f"velocity_{window_h}h_count"
        sum_col    = f"velocity_{window_h}h_sum"

        grp = df.groupby("card_id")["amount"]
        df[count_col] = grp.transform(
            lambda s: s.rolling(window_str, closed="left").count()
        ).fillna(0)
        df[sum_col] = grp.transform(
            lambda s: s.rolling(window_str, closed="left").sum()
        ).fillna(0)

    df = df.reset_index()
    return df


def _add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if CFG.features.amount_log_transform:
        df["amount_log"] = np.log1p(df["amount"])

    # Amount vs card historical mean (deviation signal)
    card_mean = df.groupby("card_id")["amount"].transform("mean")
    df["amount_vs_mean_ratio"] = df["amount"] / (card_mean + 1e-9)
    return df


def _add_merchant_risk(df: pd.DataFrame, ref_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Smoothed merchant risk score: fraud rate per merchant, shrunk toward the
    global mean when sample size is small (additive smoothing / Bayesian shrinkage).
    ref_df: training reference. Pass None to use df itself (fit+transform).
    """
    source = ref_df if ref_df is not None else df
    min_tx = CFG.features.merchant_risk_min_tx
    global_rate = source["is_fraud"].mean() if "is_fraud" in source.columns else 0.015

    if "is_fraud" in source.columns:
        merchant_stats = (
            source.groupby("merchant_id")["is_fraud"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "fraud_count", "count": "total"})
        )
        merchant_stats["risk_score"] = (
            (merchant_stats["fraud_count"] + global_rate * min_tx)
            / (merchant_stats["total"] + min_tx)
        )
        risk_map = merchant_stats["risk_score"]
    else:
        risk_map = pd.Series(dtype=float)

    df = df.copy()
    df["merchant_risk_score"] = df["merchant_id"].map(risk_map).fillna(global_rate)
    return df


# ── Public API ────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "amount_log",
    "hour_sin", "hour_cos",
    "is_weekend",
    "country_risk",
    "mcc_risk",
    "amount_vs_mean_ratio",
    "merchant_risk_score",
] + [
    f"velocity_{w}h_{agg}"
    for w in CFG.features.velocity_windows
    for agg in ("count", "sum")
]


def build_features(
    df: pd.DataFrame,
    ref_df: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full feature-engineering pipeline.

    Parameters
    ----------
    df      : raw transactions DataFrame
    ref_df  : reference DataFrame for merchant risk (use training split to avoid leakage)
    verbose : log shape at each step

    Returns
    -------
    DataFrame with FEATURE_COLUMNS added (original columns preserved).
    """
    # Normalise column names to internal schema
    df = df.rename(columns=COLUMN_MAP)

    if "is_fraud" in df.columns and df["is_fraud"].dtype == object:
        df["is_fraud"] = df["is_fraud"].map({"fraud": 1, "genuine": 0}).fillna(0).astype(int)
    # ─────────────────────────
    
    steps = [
        ("time features",     _add_time_features),
        ("risk flags",        _add_risk_flags),
        ("amount features",   _add_amount_features),
        ("velocity features", _add_velocity_features),
    ]

    for name, fn in steps:
        df = fn(df)
        if verbose:
            logger.debug("After %-20s shape=%s", name, df.shape)

    df = _add_merchant_risk(df, ref_df=ref_df)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Feature engineering incomplete — missing: {missing}")

    logger.info("Feature matrix ready  shape=%s  features=%d", df.shape, len(FEATURE_COLUMNS))
    return df


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return numpy array of FEATURE_COLUMNS (for model input)."""
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)