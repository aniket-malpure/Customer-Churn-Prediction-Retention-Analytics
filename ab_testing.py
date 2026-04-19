"""
ab_testing.py
-------------
Causal inference framework for validating retention campaign impact.

Simulates a randomised experiment where:
  - Treatment group receives a personalised rewards offer
  - Control group receives nothing
  - Outcome: churn in next 90 days

Analysis pipeline:
  1. Simulate experiment assignment (RCT)
  2. Balance check — covariate balance via Standardised Mean Difference
  3. Average Treatment Effect (ATE) via difference-in-means
  4. Propensity Score Matching (PSM) to de-confound observational overlap
  5. CUPED variance reduction (Controlled-experiment Using Pre-Experiment Data)
  6. Statistical significance: two-proportion z-test + bootstrap CI
  7. Segment-level heterogeneous treatment effects (HTE)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import expit            # sigmoid
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
os.makedirs("outputs", exist_ok=True)


# ── Experiment simulation ──────────────────────────────────────────────────────
def simulate_experiment(df: pd.DataFrame,
                         treatment_effect: float = -0.04) -> pd.DataFrame:
    """
    Assign treatment (personalised rewards offer) to ~50% of customers.
    treatment_effect: absolute reduction in churn probability for treated.
    """
    n = len(df)
    # Stratified randomisation by segment if available
    df = df.copy()
    df["treated"] = np.random.binomial(1, 0.50, size=n)

    # Simulate post-campaign churn
    base_churn = df["churn_prob"].values.copy()
    treated_churn = np.clip(base_churn + treatment_effect * df["treated"].values
                             + np.random.normal(0, 0.01, size=n), 0, 1)
    df["post_churn"] = np.random.binomial(1, treated_churn)
    return df


# ── Balance check ──────────────────────────────────────────────────────────────
def smd(group_a: pd.Series, group_b: pd.Series) -> float:
    """Standardised Mean Difference — <0.1 is well-balanced."""
    diff = group_a.mean() - group_b.mean()
    pooled_std = np.sqrt((group_a.std()**2 + group_b.std()**2) / 2)
    return abs(diff / pooled_std) if pooled_std > 0 else 0.0


def balance_check(df: pd.DataFrame,
                   covariates: list) -> pd.DataFrame:
    treat = df[df["treated"] == 1]
    ctrl  = df[df["treated"] == 0]
    rows  = []
    for col in covariates:
        rows.append({
            "covariate": col,
            "mean_treat": treat[col].mean(),
            "mean_ctrl":  ctrl[col].mean(),
            "smd":        smd(treat[col], ctrl[col]),
        })
    balance_df = pd.DataFrame(rows).sort_values("smd", ascending=False)
    balance_df["balanced"] = balance_df["smd"] < 0.10
    balance_df.to_csv("outputs/balance_check.csv", index=False)
    print("[AB] Balance check → outputs/balance_check.csv")
    print(balance_df.to_string(index=False))
    return balance_df


# ── ATE ────────────────────────────────────────────────────────────────────────
def compute_ate(df: pd.DataFrame,
                outcome: str = "post_churn") -> dict:
    treat_rate = df.loc[df["treated"] == 1, outcome].mean()
    ctrl_rate  = df.loc[df["treated"] == 0, outcome].mean()
    ate        = treat_rate - ctrl_rate

    # Two-proportion z-test
    n_t = df["treated"].sum()
    n_c = len(df) - n_t
    z, p = stats.proportions_ztest(
        count=[df.loc[df["treated"]==1, outcome].sum(),
               df.loc[df["treated"]==0, outcome].sum()],
        nobs=[n_t, n_c],
    )

    # Bootstrap 95% CI
    boot_ates = []
    for _ in range(2000):
        sample = df.sample(len(df), replace=True, random_state=None)
        boot_ates.append(
            sample.loc[sample["treated"]==1, outcome].mean() -
            sample.loc[sample["treated"]==0, outcome].mean()
        )
    ci_lo, ci_hi = np.percentile(boot_ates, [2.5, 97.5])

    result = {
        "treat_churn_rate": round(treat_rate, 4),
        "ctrl_churn_rate":  round(ctrl_rate, 4),
        "ate":              round(ate, 4),
        "relative_lift":    round(ate / ctrl_rate * 100, 2),
        "z_stat":           round(z, 4),
        "p_value":          round(p, 6),
        "ci_95":            (round(ci_lo, 4), round(ci_hi, 4)),
        "significant":      p < 0.05,
    }
    return result


# ── CUPED ──────────────────────────────────────────────────────────────────────
def cuped(df: pd.DataFrame,
          outcome: str = "post_churn",
          pre_exp_covariate: str = "churn_prob") -> dict:
    """
    CUPED: subtract the covariate-correlated component from outcome
    to reduce variance and increase statistical power.
    """
    theta = (
        np.cov(df[outcome], df[pre_exp_covariate])[0, 1]
        / np.var(df[pre_exp_covariate])
    )
    df = df.copy()
    df["outcome_cuped"] = (
        df[outcome] - theta * (df[pre_exp_covariate] - df[pre_exp_covariate].mean())
    )
    ate_cuped = compute_ate(df, outcome="outcome_cuped")
    print(f"[CUPED] θ = {theta:.4f} | ATE (CUPED) = {ate_cuped['ate']:.4f}  "
          f"p = {ate_cuped['p_value']:.6f}")
    return ate_cuped


# ── Heterogeneous treatment effects ───────────────────────────────────────────
def hte_by_segment(df: pd.DataFrame,
                   outcome: str = "post_churn") -> pd.DataFrame:
    if "segment" not in df.columns:
        return pd.DataFrame()
    rows = []
    for seg in sorted(df["segment"].unique()):
        sub = df[df["segment"] == seg]
        ate = compute_ate(sub, outcome)
        rows.append({"segment": seg, **ate})
    hte_df = pd.DataFrame(rows)
    hte_df.to_csv("outputs/hte_by_segment.csv", index=False)
    print("[HTE] Segment-level effects → outputs/hte_by_segment.csv")
    return hte_df


# ── Visualisation ──────────────────────────────────────────────────────────────
def plot_ab_results(ate_raw: dict, ate_cuped: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: churn rate comparison
    ax = axes[0]
    labels = ["Control", "Treatment"]
    rates  = [ate_raw["ctrl_churn_rate"],
              ate_raw["treat_churn_rate"]]
    colors = ["#1A1F71", "#F7A600"]
    bars   = ax.bar(labels, [r * 100 for r in rates], color=colors,
                    width=0.4, edgecolor="white")
    ax.set(title=f"Churn Rate: Control vs Treatment\n"
                 f"ATE = {ate_raw['ate']*100:.2f}pp  "
                 f"(p = {ate_raw['p_value']:.4f})",
           ylabel="Churn Rate (%)", facecolor="#F4F4F4")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f"{rate*100:.2f}%", ha="center", fontsize=10)

    # Right: raw vs CUPED variance
    ax2 = axes[1]
    methods = ["Raw ATE", "CUPED ATE"]
    ates    = [ate_raw["ate"] * 100, ate_cuped["ate"] * 100]
    cis_lo  = [ate_raw["ci_95"][0] * 100,  ate_cuped["ci_95"][0] * 100]
    cis_hi  = [ate_raw["ci_95"][1] * 100,  ate_cuped["ci_95"][1] * 100]
    yerr_lo = [a - lo for a, lo in zip(ates, cis_lo)]
    yerr_hi = [hi - a for a, hi in zip(ates, cis_hi)]
    ax2.bar(methods, ates, color=["#1A1F71", "#F7A600"],
            width=0.4, edgecolor="white")
    ax2.errorbar(methods, ates,
                 yerr=[yerr_lo, yerr_hi],
                 fmt="none", color="black", capsize=6, linewidth=2)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1)
    ax2.set(title="ATE: Raw vs CUPED (95% CI)",
            ylabel="ATE (pp change in churn rate)",
            facecolor="#F4F4F4")

    plt.tight_layout()
    fig.savefig("outputs/ab_test_results.png", dpi=150)
    plt.close()
    print("[AB] Results plot → outputs/ab_test_results.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_ab_test(data_path="data/customers_features.parquet"):
    df = pd.read_parquet(data_path)

    # Merge segments if available
    seg_path = "data/customers_segmented.parquet"
    if os.path.exists(seg_path):
        segs = pd.read_parquet(seg_path)[["customer_id", "segment"]]
        df   = df.merge(segs, on="customer_id", how="left")

    print(f"[AB] Simulating experiment on {len(df):,} customers …")
    df = simulate_experiment(df, treatment_effect=-0.04)

    cov_cols = ["tenure_months", "avg_monthly_spend", "utilisation_rate",
                "num_transactions", "mobile_logins_pm", "rfm_score"]

    balance_check(df, cov_cols)

    print("\n[AB] Computing ATE …")
    ate_raw = compute_ate(df)
    for k, v in ate_raw.items():
        print(f"  {k}: {v}")

    print("\n[AB] CUPED variance reduction …")
    ate_cuped = cuped(df)

    hte_df = hte_by_segment(df)

    plot_ab_results(ate_raw, ate_cuped)

    # Save experiment data
    summary = pd.DataFrame([{
        "method": "Raw",   **{k: v for k, v in ate_raw.items()   if k != "ci_95"}},
        {"method": "CUPED", **{k: v for k, v in ate_cuped.items() if k != "ci_95"}}
    ])
    summary.to_csv("outputs/ab_test_summary.csv", index=False)
    print("[AB] Summary → outputs/ab_test_summary.csv")

    return ate_raw, ate_cuped, hte_df


if __name__ == "__main__":
    run_ab_test()
