"""
segmentation.py
---------------
K-Means clustering to identify distinct customer cohorts
for targeted retention marketing.

Pipeline:
  1. Select RFM + engagement features
  2. Standardise + PCA to 2D for visualisation
  3. Elbow + Silhouette method to select k
  4. Fit K-Means (k=5)
  5. Profile each segment → segment_profiles.csv
  6. Output scatter plot coloured by cluster
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
SEED = 42
os.makedirs("outputs", exist_ok=True)

CLUSTER_FEATURES = [
    "tenure_months", "avg_monthly_spend", "num_transactions",
    "utilisation_rate", "mobile_logins_pm", "rfm_score",
    "num_categories", "months_active", "revolving",
]

SEGMENT_NAMES = {
    0: "High-Value Active",
    1: "Dormant Low-Spend",
    2: "Young Digital-First",
    3: "Revolving Debt-Prone",
    4: "Mature Occasional",
}

SEGMENT_COLORS = ["#1A1F71", "#F7A600", "#2196F3", "#D32F2F", "#4CAF50"]


def select_k(X_scaled: np.ndarray, k_range=range(2, 10)) -> int:
    """Elbow + Silhouette to pick optimal k."""
    inertias    = []
    silhouettes = []
    for k in k_range:
        km   = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        lbls = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, lbls, sample_size=5000))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, "o-", color="#1A1F71")
    axes[0].set(title="Elbow Method", xlabel="k", ylabel="Inertia")
    axes[1].plot(list(k_range), silhouettes, "o-", color="#F7A600")
    axes[1].set(title="Silhouette Score", xlabel="k", ylabel="Score")
    plt.tight_layout()
    fig.savefig("outputs/cluster_selection.png", dpi=150)
    plt.close()
    print("[SEG] Cluster selection plot → outputs/cluster_selection.png")

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"[SEG] Optimal k = {best_k} (silhouette = {max(silhouettes):.4f})")
    return best_k


def profile_segments(df: pd.DataFrame) -> pd.DataFrame:
    profile = (
        df.groupby("segment")
          .agg(
              n_customers    =("customer_id", "count"),
              churn_rate     =("is_churn", "mean"),
              avg_spend      =("avg_monthly_spend", "mean"),
              avg_tenure     =("tenure_months", "mean"),
              avg_util       =("utilisation_rate", "mean"),
              avg_logins     =("mobile_logins_pm", "mean"),
              avg_rfm        =("rfm_score", "mean"),
          )
          .round(4)
    )
    profile["segment_name"] = profile.index.map(SEGMENT_NAMES)
    profile["pct_of_base"]  = (profile["n_customers"] /
                                profile["n_customers"].sum() * 100).round(2)
    return profile.reset_index()


def run_segmentation(data_path="data/customers_features.parquet"):
    df = pd.read_parquet(data_path)

    X_raw    = df[CLUSTER_FEATURES].fillna(0)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # ── Select k ──────────────────────────────────────────────────────────────
    best_k = select_k(X_scaled)
    best_k = 5   # override to 5 for interpretability

    # ── Fit K-Means ──────────────────────────────────────────────────────────
    print(f"[SEG] Fitting K-Means k={best_k} …")
    km     = KMeans(n_clusters=best_k, random_state=SEED, n_init=15,
                    max_iter=400)
    labels = km.fit_predict(X_scaled)
    df["segment"] = labels

    # ── PCA for 2D scatter ────────────────────────────────────────────────────
    pca     = PCA(n_components=2, random_state=SEED)
    X_2d    = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 7))
    for seg in range(best_k):
        mask = labels == seg
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=SEGMENT_COLORS[seg], label=SEGMENT_NAMES.get(seg, seg),
            alpha=0.4, s=8, edgecolors="none",
        )
    ax.set(
        xlabel=f"PC1 ({var_exp[0]*100:.1f}% var)",
        ylabel=f"PC2 ({var_exp[1]*100:.1f}% var)",
        title="Customer Segmentation — K-Means (k=5) via PCA",
        facecolor="#F4F4F4",
    )
    ax.legend(markerscale=3, fontsize=9)
    plt.tight_layout()
    fig.savefig("outputs/customer_segments.png", dpi=150)
    plt.close()
    print("[SEG] Scatter plot → outputs/customer_segments.png")

    # ── Segment profiles ──────────────────────────────────────────────────────
    profile = profile_segments(df)
    profile.to_csv("outputs/segment_profiles.csv", index=False)
    print("[SEG] Segment profiles → outputs/segment_profiles.csv")
    print("\n", profile.to_string(index=False))

    # ── Save segmented data ───────────────────────────────────────────────────
    df.to_parquet("data/customers_segmented.parquet", index=False)
    print("[SEG] Segmented data → data/customers_segmented.parquet")

    return df, profile


if __name__ == "__main__":
    run_segmentation()
