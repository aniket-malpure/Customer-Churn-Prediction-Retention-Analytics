"""
data_generation.py
------------------
Generates synthetic credit card customer data for churn modelling.
2M+ customer records with realistic behavioural, demographic,
and product-usage signals.

Churn definition: customer made 0 transactions in last 90 days
(common in credit card analytics).
"""

import numpy as np
import pandas as pd
import os

SEED = 42
np.random.seed(SEED)

N_CUSTOMERS  = 200_000   # scaled proxy for 2M
CHURN_RATE   = 0.16      # ~16% annual churn (realistic for credit cards)

PRODUCTS     = ["Classic", "Gold", "Platinum", "Infinite"]
ACQUISITION  = ["Branch", "Online", "Partner", "Direct_Mail", "Referral"]
REGIONS      = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]


def generate_customers(n: int = N_CUSTOMERS) -> pd.DataFrame:
    np.random.seed(SEED)

    # Demographics
    tenure_months   = np.random.randint(1, 121, size=n)      # 1-10 years
    age             = np.random.randint(21, 75, size=n)
    credit_limit    = np.round(
        np.random.lognormal(mean=9.5, sigma=0.6, size=n), -2
    ).clip(1_000, 100_000)

    # Product usage
    product         = np.random.choice(PRODUCTS,     size=n,
                                        p=[0.40, 0.30, 0.20, 0.10])
    acq_channel     = np.random.choice(ACQUISITION,  size=n,
                                        p=[0.20, 0.35, 0.15, 0.20, 0.10])
    region          = np.random.choice(REGIONS,      size=n)

    # Spend behaviour (last 12 months)
    avg_monthly_spend = np.round(
        np.random.lognormal(mean=6.8, sigma=0.9, size=n), 2
    )
    spend_std         = avg_monthly_spend * np.random.uniform(0.1, 0.5, size=n)
    months_active     = np.random.randint(1, 13, size=n)
    num_categories    = np.random.randint(1, 9, size=n)       # distinct MCC cats
    num_transactions  = np.random.poisson(lam=18, size=n).clip(0, 120)
    utilisation_rate  = np.random.beta(2, 5, size=n)          # 0-1
    balance           = (credit_limit * utilisation_rate).round(2)
    revolving         = np.random.binomial(1, 0.45, size=n)   # revolves balance?

    # Engagement
    mobile_logins_pm  = np.random.poisson(lam=8, size=n).clip(0, 60)
    reward_redeemed   = np.random.binomial(1, 0.38, size=n)
    contacted_support = np.random.binomial(1, 0.12, size=n)

    # ── Churn label ────────────────────────────────────────────────────────────
    # Base churn probability influenced by multiple signals
    churn_logit = (
        - 0.03 * tenure_months
        - 0.5  * (months_active / 12)
        - 0.8  * utilisation_rate
        - 0.4  * (num_transactions / 60)
        + 0.6  * (1 - reward_redeemed)
        + 0.3  * contacted_support
        - 0.2  * (mobile_logins_pm / 20)
        + np.random.normal(0, 0.5, size=n)   # noise
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    # Re-scale to target CHURN_RATE
    churn_prob = churn_prob / churn_prob.mean() * CHURN_RATE
    churn_prob = churn_prob.clip(0, 0.99)
    is_churn   = np.random.binomial(1, churn_prob)

    df = pd.DataFrame({
        "customer_id":          range(1, n + 1),
        "tenure_months":        tenure_months,
        "age":                  age,
        "credit_limit":         credit_limit,
        "product":              product,
        "acquisition_channel":  acq_channel,
        "region":               region,
        "avg_monthly_spend":    avg_monthly_spend,
        "spend_std":            spend_std.round(2),
        "months_active":        months_active,
        "num_categories":       num_categories,
        "num_transactions":     num_transactions,
        "utilisation_rate":     utilisation_rate.round(4),
        "balance":              balance,
        "revolving":            revolving,
        "mobile_logins_pm":     mobile_logins_pm,
        "reward_redeemed":      reward_redeemed,
        "contacted_support":    contacted_support,
        "churn_prob":           churn_prob.round(4),
        "is_churn":             is_churn,
    })

    return df


if __name__ == "__main__":
    print(f"[INFO] Generating {N_CUSTOMERS:,} customer records …")
    df = generate_customers()
    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/customers_raw.parquet", index=False)
    n_churn = df["is_churn"].sum()
    print(f"[INFO] Saved → data/customers_raw.parquet")
    print(f"[INFO] Rows: {len(df):,}  |  Churned: {n_churn:,}  "
          f"({100*n_churn/len(df):.2f}%)")
    print(df.describe().to_string())
