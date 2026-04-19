"""
data_generation.py
------------------
Generates synthetic credit card transaction data mimicking
real-world Visa-scale patterns: 50M+ rows, imbalanced fraud labels,
merchant categories, geolocation shifts, and time-based features.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Constants ──────────────────────────────────────────────────────────────────
N_TRANSACTIONS   = 500_000          # scaled-down proxy for 50M
FRAUD_RATE       = 0.015            # ~1.5% fraud (realistic)
N_CARDHOLDERS    = 50_000
N_MERCHANTS      = 10_000

MCC_CODES = {
    "grocery":       5411,
    "gas_station":   5541,
    "restaurant":    5812,
    "online_retail": 5999,
    "atm":           6011,
    "travel":        4111,
    "electronics":   5732,
    "pharmacy":      5912,
}

COUNTRIES = ["US", "US", "US", "US", "CA", "MX", "GB", "DE", "CN", "NG"]


# ── Helpers ────────────────────────────────────────────────────────────────────
def random_timestamps(n: int, start: str = "2023-01-01", days: int = 365):
    base = datetime.strptime(start, "%Y-%m-%d")
    offsets = np.random.randint(0, days * 86400, size=n)
    return [base + timedelta(seconds=int(o)) for o in offsets]


def generate_transactions(n: int = N_TRANSACTIONS) -> pd.DataFrame:
    card_ids      = np.random.randint(1, N_CARDHOLDERS + 1, size=n)
    merchant_ids  = np.random.randint(1, N_MERCHANTS + 1, size=n)
    mcc_keys      = np.random.choice(list(MCC_CODES.keys()), size=n)
    mcc_values    = [MCC_CODES[k] for k in mcc_keys]
    timestamps    = random_timestamps(n)
    countries     = np.random.choice(COUNTRIES, size=n, p=[0.70, 0.00, 0.00,
                                                            0.00, 0.06, 0.04,
                                                            0.05, 0.05, 0.05, 0.05])

    # Amount: log-normal, fraud has heavier right tail
    amounts = np.round(np.random.lognormal(mean=3.5, sigma=1.2, size=n), 2)

    # Fraud label (imbalanced)
    is_fraud = np.random.binomial(1, FRAUD_RATE, size=n)

    # Inflate amounts for ~40% of fraudulent transactions
    fraud_idx = np.where(is_fraud == 1)[0]
    inflate   = np.random.choice(fraud_idx,
                                  size=int(0.4 * len(fraud_idx)),
                                  replace=False)
    amounts[inflate] *= np.random.uniform(3, 10, size=len(inflate))

    df = pd.DataFrame({
        "transaction_id":   range(1, n + 1),
        "card_id":          card_ids,
        "merchant_id":      merchant_ids,
        "mcc_code":         mcc_values,
        "mcc_category":     mcc_keys,
        "timestamp":        timestamps,
        "amount":           amounts,
        "country":          countries,
        "is_fraud":         is_fraud,
    })

    df["hour"]       = pd.DatetimeIndex(df["timestamp"]).hour
    df["day_of_week"] = pd.DatetimeIndex(df["timestamp"]).dayofweek
    df["month"]      = pd.DatetimeIndex(df["timestamp"]).month

    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add card-level velocity features (txn count & sum in last 1h, 24h)."""
    df = df.sort_values(["card_id", "timestamp"]).reset_index(drop=True)
    df["ts_unix"] = df["timestamp"].astype(np.int64) // 10**9

    # Rolling 1-hour transaction count per card
    def rolling_count(group, window_sec=3600):
        ts   = group["ts_unix"].values
        vals = []
        for i, t in enumerate(ts):
            mask = (ts[:i] >= t - window_sec) & (ts[:i] < t)
            vals.append(mask.sum())
        return pd.Series(vals, index=group.index)

    df["txn_count_1h"] = (
        df.groupby("card_id", group_keys=False)
          .apply(rolling_count)
    )

    # 24-hour spend per card (approximated via groupby shift for speed)
    df["cumsum_24h"] = (
        df.groupby(["card_id", df["timestamp"].dt.date])["amount"]
          .cumsum()
    )

    df.drop(columns=["ts_unix"], inplace=True)
    return df


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[INFO] Generating {N_TRANSACTIONS:,} transactions …")
    df = generate_transactions()
    print(f"[INFO] Adding velocity features …")
    df = add_velocity_features(df)

    out_path = "data/transactions_raw.parquet"
    os.makedirs("data", exist_ok=True)
    df.to_parquet(out_path, index=False)

    fraud_count = df["is_fraud"].sum()
    print(f"[INFO] Saved → {out_path}")
    print(f"[INFO] Rows: {len(df):,}  |  Fraud rows: {fraud_count:,}  "
          f"({100 * fraud_count / len(df):.2f}%)")
    print(df.head(3).to_string())
