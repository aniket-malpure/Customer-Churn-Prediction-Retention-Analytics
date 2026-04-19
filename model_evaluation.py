"""
model_evaluation.py
--------------------
Post-training evaluation module.
  1. Confusion matrix, ROC curve, Precision-Recall curve
  2. SHAP global + local explanations
  3. False-positive analysis by MCC category
  4. Outputs saved to outputs/ for Tableau / reporting
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay,
)

from feature_engineering import FEATURE_COLS, TARGET_COL, build_features

os.makedirs("outputs", exist_ok=True)


# ── Load artefacts ─────────────────────────────────────────────────────────────
def load_artefacts():
    model     = joblib.load("models/xgb_fraud_model.pkl")
    threshold = joblib.load("models/best_threshold.pkl")
    features  = joblib.load("models/feature_cols.pkl")
    return model, threshold, features


# ── ROC curve ─────────────────────────────────────────────────────────────────
def plot_roc(y_true, y_proba, ax=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"XGBoost (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve — Fraud Detection")
    ax.legend()
    return ax


# ── Precision-Recall curve ─────────────────────────────────────────────────────
def plot_pr(y_true, y_proba, threshold, ax=None):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"AP = {pr_auc:.3f}")
    # Mark operating threshold
    idx = np.argmin(np.abs(thresholds - threshold))
    ax.scatter(recall[idx], precision[idx], s=80, color="red",
               zorder=5, label=f"Threshold = {threshold:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curve — Fraud Detection")
    ax.legend()
    return ax


# ── Confusion matrix ──────────────────────────────────────────────────────────
def plot_confusion(y_true, y_pred):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    return fig


# ── SHAP explanations ─────────────────────────────────────────────────────────
def shap_analysis(model, X_sample: pd.DataFrame, n_summary=500):
    print("[SHAP] Computing SHAP values …")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample.head(n_summary))

    # Summary plot
    fig_summary, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample.head(n_summary),
                      plot_type="bar", show=False)
    plt.tight_layout()
    fig_summary.savefig("outputs/shap_summary.png", dpi=150)
    plt.close()
    print("[SHAP] Summary plot → outputs/shap_summary.png")

    # Local force plot for the top fraud example
    fraud_indices = X_sample.index[
        model.predict_proba(X_sample)[:, 1] > 0.8
    ]
    if len(fraud_indices):
        idx = fraud_indices[0]
        shap.force_plot(
            explainer.expected_value,
            shap_values[X_sample.index.get_loc(idx)],
            X_sample.loc[idx],
            matplotlib=True,
            show=False,
        )
        plt.savefig("outputs/shap_local_force.png", dpi=150,
                    bbox_inches="tight")
        plt.close()
        print("[SHAP] Local force plot → outputs/shap_local_force.png")

    return shap_values


# ── False-positive breakdown ──────────────────────────────────────────────────
def fp_analysis(df: pd.DataFrame, y_pred: np.ndarray, y_true: np.ndarray):
    """Which MCC categories generate the most false positives?"""
    analysis = df.copy()
    analysis["y_true"] = y_true
    analysis["y_pred"] = y_pred
    analysis["fp"] = ((analysis["y_pred"] == 1) &
                      (analysis["y_true"] == 0)).astype(int)

    fp_by_mcc = (
        analysis.groupby("mcc_category")
                .agg(fp_count=("fp", "sum"),
                     txn_count=("fp", "count"))
                .assign(fp_rate=lambda x: x["fp_count"] / x["txn_count"])
                .sort_values("fp_rate", ascending=False)
    )
    fp_by_mcc.to_csv("outputs/false_positive_by_mcc.csv")
    print("[FP] False-positive breakdown → outputs/false_positive_by_mcc.csv")
    print(fp_by_mcc.to_string())
    return fp_by_mcc


# ── Main ──────────────────────────────────────────────────────────────────────
def evaluate():
    model, threshold, features = load_artefacts()

    raw  = pd.read_parquet("data/transactions_raw.parquet")
    feat = pd.read_parquet("data/transactions_features.parquet")
    X    = feat[features].fillna(0)
    y    = feat[TARGET_COL]

    # Use the held-out 15% test portion (last rows by approximation)
    n_test  = int(0.15 * len(X))
    X_test  = X.iloc[-n_test:]
    y_test  = y.iloc[-n_test:]
    raw_test = raw.iloc[-n_test:]

    proba  = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_roc(y_test, proba, ax=axes[0])
    plot_pr(y_test, proba, threshold, ax=axes[1])
    plt.tight_layout()
    fig.savefig("outputs/roc_pr_curves.png", dpi=150)
    plt.close()
    print("[EVAL] ROC / PR curves → outputs/roc_pr_curves.png")

    fig_cm = plot_confusion(y_test, y_pred)
    fig_cm.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()
    print("[EVAL] Confusion matrix → outputs/confusion_matrix.png")

    # SHAP
    shap_analysis(model, X_test)

    # FP breakdown
    fp_analysis(raw_test.reset_index(drop=True),
                y_pred, y_test.values)

    # Save scored test data for Tableau
    out_df = raw_test.copy().reset_index(drop=True)
    out_df["fraud_proba"] = proba
    out_df["fraud_flag"]  = y_pred
    out_df.to_csv("outputs/test_scored.csv", index=False)
    print("[EVAL] Scored test data → outputs/test_scored.csv")


if __name__ == "__main__":
    evaluate()
