"""
dashboard.py
-------------
Generates a static 4-panel analytics dashboard (PNG) that mirrors
the kind of Tableau view a fraud risk team would consume.

Panels:
  1. Fraud rate by MCC category (bar)
  2. Fraud amount distribution vs legit (KDE overlay)
  3. Fraud rate by hour of day (line)
  4. Top-10 high-risk merchants (horizontal bar)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

os.makedirs("outputs", exist_ok=True)

VISA_BLUE  = "#1A1F71"
VISA_GOLD  = "#F7A600"
ALERT_RED  = "#D32F2F"
LIGHT_GREY = "#F4F4F4"
MID_GREY   = "#AAAAAA"


def load(path="data/transactions_features.parquet"):
    df = pd.read_parquet(path)
    scored_path = "outputs/test_scored.csv"
    if os.path.exists(scored_path):
        scored = pd.read_csv(scored_path)
        return df, scored
    return df, None


# ── Panel helpers ──────────────────────────────────────────────────────────────
def panel_fraud_by_mcc(ax, df):
    rates = (
        df.groupby("mcc_category")["is_fraud"]
          .mean()
          .sort_values(ascending=True)
          .mul(100)
    )
    colors = [ALERT_RED if r > 2 else VISA_BLUE for r in rates]
    bars = ax.barh(rates.index, rates.values, color=colors, edgecolor="white")
    ax.set_xlabel("Fraud Rate (%)")
    ax.set_title("Fraud Rate by Merchant Category", fontweight="bold", pad=10)
    ax.set_facecolor(LIGHT_GREY)
    for bar, val in zip(bars, rates.values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=8)
    ax.axvline(rates.mean(), color=VISA_GOLD, linestyle="--",
               linewidth=1.2, label=f"Avg {rates.mean():.2f}%")
    ax.legend(fontsize=8)


def panel_amount_kde(ax, df):
    fraud  = np.log1p(df.loc[df["is_fraud"] == 1, "amount"])
    legit  = np.log1p(df.loc[df["is_fraud"] == 0, "amount"])

    for vals, color, label in [(legit, VISA_BLUE, "Legit"),
                                (fraud, ALERT_RED, "Fraud")]:
        kde = gaussian_kde(vals, bw_method=0.3)
        x   = np.linspace(vals.min(), vals.max(), 300)
        ax.fill_between(x, kde(x), alpha=0.35, color=color)
        ax.plot(x, kde(x), color=color, lw=2, label=label)

    ax.set_xlabel("log(Amount + 1)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction Amount Distribution (log scale)",
                 fontweight="bold", pad=10)
    ax.set_facecolor(LIGHT_GREY)
    ax.legend()


def panel_hourly_fraud(ax, df):
    hourly = df.groupby("hour")["is_fraud"].mean().mul(100)
    ax.plot(hourly.index, hourly.values, color=VISA_BLUE, lw=2, marker="o",
            markersize=4)
    ax.fill_between(hourly.index, hourly.values, alpha=0.15, color=VISA_BLUE)
    ax.axhspan(hourly.quantile(0.75), hourly.max(),
               alpha=0.12, color=ALERT_RED, label="High-risk window")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("Fraud Rate by Hour of Day", fontweight="bold", pad=10)
    ax.set_xticks(range(0, 24, 2))
    ax.set_facecolor(LIGHT_GREY)
    ax.legend(fontsize=8)


def panel_top_merchants(ax, df):
    top_merch = (
        df.groupby("merchant_id")
          .agg(fraud_rate=("is_fraud", "mean"),
               vol=("is_fraud", "count"))
          .query("vol >= 50")
          .nlargest(10, "fraud_rate")
    )
    top_merch = top_merch.reset_index()
    labels    = [f"Merchant {m}" for m in top_merch["merchant_id"]]
    colors    = [ALERT_RED if r > 0.05 else VISA_GOLD
                 for r in top_merch["fraud_rate"]]

    ax.barh(labels, top_merch["fraud_rate"] * 100, color=colors,
            edgecolor="white")
    ax.set_xlabel("Fraud Rate (%)")
    ax.set_title("Top 10 High-Risk Merchants", fontweight="bold", pad=10)
    ax.set_facecolor(LIGHT_GREY)
    legend_elems = [
        Line2D([0], [0], color=ALERT_RED, lw=6, label="> 5% fraud"),
        Line2D([0], [0], color=VISA_GOLD,  lw=6, label="≤ 5% fraud"),
    ]
    ax.legend(handles=legend_elems, fontsize=8)


# ── Main ──────────────────────────────────────────────────────────────────────
def build_dashboard():
    df, scored = load()
    if scored is not None:
        # Merge scored probabilities back for richer visuals
        df = df.merge(
            scored[["transaction_id", "fraud_proba"]],
            on="transaction_id", how="left"
        )

    fig = plt.figure(figsize=(18, 12), facecolor="white")
    fig.suptitle(
        "Credit Card Fraud Detection — Risk Analytics Dashboard",
        fontsize=16, fontweight="bold", color=VISA_BLUE, y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.38, wspace=0.28,
                           left=0.07, right=0.97,
                           top=0.92, bottom=0.07)

    panel_fraud_by_mcc(fig.add_subplot(gs[0, 0]), df)
    panel_amount_kde(fig.add_subplot(gs[0, 1]), df)
    panel_hourly_fraud(fig.add_subplot(gs[1, 0]), df)
    panel_top_merchants(fig.add_subplot(gs[1, 1]), df)

    out = "outputs/fraud_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[DASHBOARD] Saved → {out}")


if __name__ == "__main__":
    build_dashboard()
