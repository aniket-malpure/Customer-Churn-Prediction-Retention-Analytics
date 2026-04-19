"""
dashboard.py
-------------
5-panel churn analytics dashboard.

Panels:
  1. Churn rate by customer segment (bar)
  2. Churn probability distribution (KDE)
  3. Feature importance from trained model
  4. A/B test results (lift waterfall)
  5. Churn rate by product tier
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib

os.makedirs("outputs", exist_ok=True)

VISA_BLUE  = "#1A1F71"
VISA_GOLD  = "#F7A600"
ALERT_RED  = "#D32F2F"
LIGHT_GREY = "#F4F4F4"

SEGMENT_NAMES = {
    0: "High-Value",
    1: "Dormant",
    2: "Digital-First",
    3: "Debt-Prone",
    4: "Occasional",
}


def load_data():
    df_feat = pd.read_parquet("data/customers_features.parquet")
    df_raw  = pd.read_parquet("data/customers_raw.parquet")
    df      = df_feat.merge(df_raw[["customer_id", "product",
                                    "avg_monthly_spend", "tenure_months"]],
                             on="customer_id", how="left",
                             suffixes=("", "_raw"))
    seg_path = "data/customers_segmented.parquet"
    if os.path.exists(seg_path):
        segs = pd.read_parquet(seg_path)[["customer_id", "segment"]]
        df   = df.merge(segs, on="customer_id", how="left")
    return df


# ── Panel 1: Churn by segment ──────────────────────────────────────────────────
def panel_segment_churn(ax, df):
    if "segment" not in df.columns:
        ax.text(0.5, 0.5, "No segment data", ha="center", transform=ax.transAxes)
        return
    rates = (
        df.groupby("segment")["is_churn"].mean()
          .rename(index=SEGMENT_NAMES)
          .sort_values(ascending=True)
          .mul(100)
    )
    colors = [ALERT_RED if r > rates.mean() else VISA_BLUE for r in rates]
    bars   = ax.barh(rates.index, rates.values, color=colors, edgecolor="white")
    ax.axvline(rates.mean(), color=VISA_GOLD, lw=1.5, ls="--",
               label=f"Avg {rates.mean():.1f}%")
    ax.set(xlabel="Churn Rate (%)", title="Churn Rate by Segment",
           facecolor=LIGHT_GREY)
    ax.legend(fontsize=8)
    for bar, val in zip(bars, rates.values):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=8)


# ── Panel 2: Churn probability KDE ────────────────────────────────────────────
def panel_churn_kde(ax, df):
    from scipy.stats import gaussian_kde
    churned  = df.loc[df["is_churn"] == 1, "churn_prob"]
    retained = df.loc[df["is_churn"] == 0, "churn_prob"]
    for vals, color, label in [(retained, VISA_BLUE, "Retained"),
                                (churned,  ALERT_RED, "Churned")]:
        kde = gaussian_kde(vals, bw_method=0.2)
        x   = np.linspace(0, 1, 300)
        ax.fill_between(x, kde(x), alpha=0.3, color=color)
        ax.plot(x, kde(x), color=color, lw=2, label=label)
    ax.set(xlabel="Churn Probability", ylabel="Density",
           title="Churn Probability Distribution", facecolor=LIGHT_GREY)
    ax.legend()


# ── Panel 3: Feature importance ───────────────────────────────────────────────
def panel_feature_importance(ax):
    model_path = "models/churn_model.pkl"
    feat_path  = "models/churn_feature_cols.pkl"
    if not os.path.exists(model_path):
        ax.text(0.5, 0.5, "Train model first", ha="center",
                transform=ax.transAxes)
        return
    model    = joblib.load(model_path)
    features = joblib.load(feat_path)
    # Works for tree-based models with feature_importances_
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if not hasattr(clf, "feature_importances_"):
        ax.text(0.5, 0.5, "No feature_importances_", ha="center",
                transform=ax.transAxes)
        return
    imp = pd.Series(clf.feature_importances_, index=features).nlargest(12)
    imp = imp.sort_values()
    ax.barh(imp.index, imp.values, color=VISA_BLUE, edgecolor="white")
    ax.set(title="Top 12 Feature Importances", xlabel="Importance",
           facecolor=LIGHT_GREY)


# ── Panel 4: A/B test lift ────────────────────────────────────────────────────
def panel_ab_lift(ax):
    ab_path = "outputs/ab_test_summary.csv"
    if not os.path.exists(ab_path):
        ax.text(0.5, 0.5, "Run ab_testing.py first", ha="center",
                transform=ax.transAxes)
        return
    ab  = pd.read_csv(ab_path)
    raw = ab[ab["method"] == "Raw"].iloc[0]
    ctrl  = raw["ctrl_churn_rate"] * 100
    treat = raw["treat_churn_rate"] * 100
    lift  = ctrl - treat

    categories = ["Baseline\nChurn", "Campaign\nLift", "Post-Campaign\nChurn"]
    values     = [ctrl, -lift, treat]
    colors     = [VISA_BLUE, VISA_GOLD, ALERT_RED]
    bars       = ax.bar(categories, [ctrl, lift, treat], color=colors,
                        edgecolor="white", width=0.5)

    # Arrow annotation
    ax.annotate(f"↓ {lift:.2f}pp\nchurn reduction",
                xy=(1, treat + lift/2), xytext=(1.5, (ctrl + treat)/2),
                fontsize=9, color=VISA_GOLD,
                arrowprops=dict(arrowstyle="->", color=VISA_GOLD))
    ax.set(title=f"A/B Test: Campaign Impact\n"
                 f"(p = {raw['p_value']:.4f})",
           ylabel="Churn Rate (%)", facecolor=LIGHT_GREY)


# ── Panel 5: Churn by product ─────────────────────────────────────────────────
def panel_product_churn(ax, df):
    product_col = "product" if "product" in df.columns else None
    if not product_col:
        ax.text(0.5, 0.5, "No product data", ha="center",
                transform=ax.transAxes)
        return
    order = ["Classic", "Gold", "Platinum", "Infinite"]
    rates = (
        df.groupby(product_col)["is_churn"].mean().reindex(order).mul(100)
    )
    colors = [VISA_GOLD if r == rates.max() else VISA_BLUE for r in rates]
    ax.bar(rates.index, rates.values, color=colors, edgecolor="white",
           width=0.5)
    ax.set(title="Churn Rate by Product Tier",
           ylabel="Churn Rate (%)", facecolor=LIGHT_GREY)
    for i, (prod, val) in enumerate(rates.items()):
        ax.text(i, val + 0.1, f"{val:.1f}%", ha="center", fontsize=9)


# ── Main ──────────────────────────────────────────────────────────────────────
def build_dashboard():
    df = load_data()

    fig = plt.figure(figsize=(20, 14), facecolor="white")
    fig.suptitle(
        "Customer Churn Prediction — Retention Analytics Dashboard",
        fontsize=16, fontweight="bold", color=VISA_BLUE, y=0.98,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.40, wspace=0.30,
                            left=0.07, right=0.97,
                            top=0.92, bottom=0.07)

    panel_segment_churn(fig.add_subplot(gs[0, 0]), df)
    panel_churn_kde(fig.add_subplot(gs[0, 1]), df)
    panel_feature_importance(fig.add_subplot(gs[0, 2]))
    panel_ab_lift(fig.add_subplot(gs[1, 0]))
    panel_product_churn(fig.add_subplot(gs[1, 1]), df)

    # Summary KPI panel
    ax_kpi = fig.add_subplot(gs[1, 2])
    ax_kpi.axis("off")
    churn_rate = df["is_churn"].mean() * 100
    n_at_risk  = (df["churn_prob"] > 0.5).sum()
    kpis = [
        ("Total Customers",    f"{len(df):,}"),
        ("Overall Churn Rate", f"{churn_rate:.1f}%"),
        ("At-Risk Customers",  f"{n_at_risk:,}"),
        ("Segments",           "5"),
    ]
    ax_kpi.set_facecolor(VISA_BLUE)
    for i, (label, val) in enumerate(kpis):
        y = 0.85 - i * 0.22
        ax_kpi.text(0.5, y,      val,   ha="center", va="center",
                    fontsize=20, fontweight="bold", color=VISA_GOLD,
                    transform=ax_kpi.transAxes)
        ax_kpi.text(0.5, y-0.08, label, ha="center", va="center",
                    fontsize=9, color="white",
                    transform=ax_kpi.transAxes)
    ax_kpi.set_title("Key Metrics", fontweight="bold", color=VISA_BLUE, pad=10)

    out = "outputs/churn_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[DASHBOARD] Saved → {out}")


if __name__ == "__main__":
    build_dashboard()
