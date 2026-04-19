"""
feature_engineering.py
-----------------------
Builds an ML-ready feature set from raw customer data.
Covers:
  - Spend decay velocity (engagement dropping off)
  - Credit utilisation bands
  - Recency-Frequency-Monetary (RFM) proxies
  - Encoded categoricals
  - Interaction features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os


CATEGORICAL_COLS = ["product", "acquisition_channel", "region"]
TARGET_COL       = "is_churn"


def rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Proxy RFM: Recency = months_active inverse, Frequency = txns, Monetary = spend."""
    df["rfm_recency"]   = 1 / (df["months_active"].clip(lower=1))
    df["rfm_frequency"] = df["num_transactions"]
    df["rfm_monetary"]  = df["avg_monthly_spend"] * df["months_active"]
    df["rfm_score"]     = (
        (1 - df["rfm_recency"] / df["rfm_recency"].max()) * 0.3
        + (df["rfm_frequency"] / df["rfm_frequency"].max()) * 0.4
        + (df["rfm_monetary"]  / df["rfm_monetary"].max())  * 0.3
    )
    return df


def spend_decay(df: pd.DataFrame) -> pd.DataFrame:
    """Coefficient of variation of monthly spend as engagement signal."""
    df["spend_cv"] = (df["spend_std"] / df["avg_monthly_spend"].clip(lower=1)).round(4)
    return df


def utilisation_bands(df: pd.DataFrame) -> pd.DataFrame:
    bins   = [-0.001, 0.1, 0.3, 0.6, 0.9, 1.01]
    labels = [0, 1, 2, 3, 4]    # 0 = almost zero, 4 = maxed out
    df["util_band"] = pd.cut(df["utilisation_rate"], bins=bins,
                              labels=labels).astype(int)
    return df


def interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross terms that signal disengagement."""
    df["low_spend_long_tenure"]  = (
        (df["avg_monthly_spend"] < df["avg_monthly_spend"].median()) &
        (df["tenure_months"] > 24)
    ).astype(int)

    df["high_util_no_rewards"]   = (
        (df["utilisation_rate"] > 0.6) &
        (df["reward_redeemed"] == 0)
    ).astype(int)

    df["inactive_mobile"]        = (
        (df["mobile_logins_pm"] < 2) &
        (df["months_active"] < 6)
    ).astype(int)

    df["spend_per_category"]     = (
        df["avg_monthly_spend"] / df["num_categories"].clip(lower=1)
    )
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    le_map = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        le_map[col] = le
    return df, le_map


FEATURE_COLS = [
    "tenure_months", "age", "credit_limit",
    "avg_monthly_spend", "spend_std", "spend_cv",
    "months_active", "num_categories", "num_transactions",
    "utilisation_rate", "util_band", "balance", "revolving",
    "mobile_logins_pm", "reward_redeemed", "contacted_support",
    "rfm_recency", "rfm_frequency", "rfm_monetary", "rfm_score",
    "low_spend_long_tenure", "high_util_no_rewards", "inactive_mobile",
    "spend_per_category",
    "product_enc", "acquisition_channel_enc", "region_enc",
]


def build_features(df: pd.DataFrame):
    print("[FE] RFM features …")
    df = rfm_features(df)
    print("[FE] Spend decay …")
    df = spend_decay(df)
    print("[FE] Utilisation bands …")
    df = utilisation_bands(df)
    print("[FE] Interaction features …")
    df = interaction_features(df)
    print("[FE] Encoding categoricals …")
    df, le_map = encode_categoricals(df)
    print(f"[FE] Feature matrix shape: {df.shape}")
    return df, le_map


if __name__ == "__main__":
    raw_path = "data/customers_raw.parquet"
    if not os.path.exists(raw_path):
        raise FileNotFoundError("Run data_generation.py first.")
    df = pd.read_parquet(raw_path)
    df, _ = build_features(df)
    out_path = "data/customers_features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[INFO] Saved → {out_path}")
    print(f"[INFO] Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
