"""
feature_engineering.py
-----------------------
Transforms raw transaction data into an ML-ready feature matrix.
Covers:
  - Cardholder behavioural aggregates
  - Merchant-level risk scores
  - Time-based cyclical encodings
  - Cross-border / high-risk MCC flags
  - Amount z-score relative to cardholder history
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os


# ── Risk lookups ───────────────────────────────────────────────────────────────
HIGH_RISK_MCC = {6011, 5999, 4111}          # ATM, online retail, travel
HIGH_RISK_COUNTRIES = {"NG", "CN", "MX"}


def cyclical_encode(series: pd.Series, max_val: int):
    """Sin/cos encoding preserves circular nature of hour/day."""
    sin = np.sin(2 * np.pi * series / max_val)
    cos = np.cos(2 * np.pi * series / max_val)
    return sin, cos


def cardholder_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-card rolling statistics over the full history window."""
    agg = (
        df.groupby("card_id")["amount"]
          .agg(["mean", "std", "median", "max", "count"])
          .rename(columns={
              "mean":   "card_avg_amount",
              "std":    "card_std_amount",
              "median": "card_median_amount",
              "max":    "card_max_amount",
              "count":  "card_txn_total",
          })
    )
    return df.merge(agg, on="card_id", how="left")


def merchant_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Merchant-level fraud rate used as a risk proxy feature."""
    merch_fraud = (
        df.groupby("merchant_id")["is_fraud"]
          .agg(fraud_rate="mean", txn_vol="count")
          .reset_index()
    )
    merch_fraud["merchant_risk"] = (
        merch_fraud["fraud_rate"] * np.log1p(merch_fraud["txn_vol"])
    )
    return df.merge(merch_fraud[["merchant_id", "merchant_risk"]],
                    on="merchant_id", how="left")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[FE] Cardholder aggregates …")
    df = cardholder_aggregates(df)

    print("[FE] Merchant risk scores …")
    df = merchant_risk_score(df)

    # Amount z-score relative to cardholder mean
    df["amount_zscore"] = (
        (df["amount"] - df["card_avg_amount"])
        / (df["card_std_amount"].replace(0, 1))
    )

    # Cyclical time encodings
    df["hour_sin"], df["hour_cos"] = cyclical_encode(df["hour"], 24)
    df["dow_sin"],  df["dow_cos"]  = cyclical_encode(df["day_of_week"], 7)

    # Binary flags
    df["is_high_risk_mcc"]     = df["mcc_code"].isin(HIGH_RISK_MCC).astype(int)
    df["is_high_risk_country"] = df["country"].isin(HIGH_RISK_COUNTRIES).astype(int)
    df["is_domestic"]          = (df["country"] == "US").astype(int)
    df["is_weekend"]           = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]             = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    # Amount buckets
    df["amount_log"] = np.log1p(df["amount"])
    df["is_large_txn"] = (df["amount"] > df["card_avg_amount"] * 3).astype(int)

    # Velocity ratio
    df["velocity_ratio"] = df["txn_count_1h"] / (df["card_txn_total"] / 24).clip(lower=0.01)

    # Fill NAs from std=0 cards
    df["card_std_amount"].fillna(0, inplace=True)

    print(f"[FE] Feature matrix shape: {df.shape}")
    return df


FEATURE_COLS = [
    "amount", "amount_log", "amount_zscore",
    "card_avg_amount", "card_std_amount", "card_median_amount",
    "card_max_amount", "card_txn_total",
    "merchant_risk",
    "txn_count_1h", "cumsum_24h", "velocity_ratio",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_high_risk_mcc", "is_high_risk_country",
    "is_domestic", "is_weekend", "is_night",
    "is_large_txn", "month",
]

TARGET_COL = "is_fraud"


if __name__ == "__main__":
    raw_path = "data/transactions_raw.parquet"
    if not os.path.exists(raw_path):
        raise FileNotFoundError("Run data_generation.py first.")

    df = pd.read_parquet(raw_path)
    df = build_features(df)

    out_path = "data/transactions_features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[INFO] Saved → {out_path}")
    print(f"[INFO] Features used: {FEATURE_COLS}")
