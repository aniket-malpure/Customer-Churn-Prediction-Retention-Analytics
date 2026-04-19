"""
model_training.py
-----------------
Trains an XGBoost fraud-detection classifier on the engineered features.
Handles severe class imbalance via SMOTE oversampling.
Saves the best model + threshold to disk.

Key design decisions:
  - SMOTE applied only on the TRAIN split (no leakage)
  - scale_pos_weight as a second imbalance lever
  - Threshold tuned to maximise F2 (recall-heavy) on validation set
  - MLflow used for experiment tracking
"""

import os
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    classification_report, average_precision_score,
    fbeta_score,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

from feature_engineering import FEATURE_COLS, TARGET_COL

warnings.filterwarnings("ignore")
SEED = 42

# ── Hyperparameters ────────────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":      600,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  5,
    "gamma":             1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "scale_pos_weight":  50,     # rough inverse of fraud ratio
    "use_label_encoder": False,
    "eval_metric":       "aucpr",
    "random_state":      SEED,
    "n_jobs":            -1,
}


def load_data(path: str = "data/transactions_features.parquet"):
    df = pd.read_parquet(path)
    X  = df[FEATURE_COLS].fillna(0)
    y  = df[TARGET_COL]
    return X, y


def tune_threshold(model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """Return the probability threshold that maximises F2 on val set."""
    proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, proba)
    f2_scores = [
        fbeta_score(y_val, (proba >= t).astype(int), beta=2)
        for t in thresholds
    ]
    best_threshold = float(thresholds[np.argmax(f2_scores)])
    print(f"[THRESH] Best threshold (max F2): {best_threshold:.4f}  "
          f"| F2: {max(f2_scores):.4f}")
    return best_threshold


def train(data_path: str = "data/transactions_features.parquet"):
    X, y = load_data(data_path)
    print(f"[DATA] Shape: {X.shape}  |  Fraud rate: {y.mean():.4f}")

    # ── Train / val / test split ──────────────────────────────────────────────
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.15, random_state=SEED, stratify=y_trainval
    )

    print(f"[SPLIT] Train: {len(X_train):,}  Val: {len(X_val):,}  "
          f"Test: {len(X_test):,}")

    # ── SMOTE on train only ───────────────────────────────────────────────────
    print("[SMOTE] Resampling training set …")
    smote = SMOTE(sampling_strategy=0.15, random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"[SMOTE] Post-resample fraud rate: {y_train_res.mean():.4f}")

    # ── MLflow tracking ───────────────────────────────────────────────────────
    mlflow.set_experiment("fraud_detection")
    with mlflow.start_run(run_name="xgboost_smote"):
        mlflow.log_params(XGB_PARAMS)

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_train_res, y_train_res,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        # ── Threshold tuning ──────────────────────────────────────────────────
        best_threshold = tune_threshold(model, X_val, y_val)
        mlflow.log_metric("best_threshold", best_threshold)

        # ── Test evaluation ───────────────────────────────────────────────────
        proba_test  = model.predict_proba(X_test)[:, 1]
        y_pred_test = (proba_test >= best_threshold).astype(int)

        auc    = roc_auc_score(y_test, proba_test)
        pr_auc = average_precision_score(y_test, proba_test)
        f2     = fbeta_score(y_test, y_pred_test, beta=2)

        mlflow.log_metrics({"roc_auc": auc, "pr_auc": pr_auc, "f2_score": f2})
        mlflow.xgboost.log_model(model, "xgboost_fraud_model")

        print("\n" + "=" * 60)
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  PR-AUC   : {pr_auc:.4f}")
        print(f"  F2 Score : {f2:.4f}")
        print("=" * 60)
        print(classification_report(y_test, y_pred_test,
                                    target_names=["Legit", "Fraud"]))

        # ── Persist artefacts ─────────────────────────────────────────────────
        os.makedirs("models", exist_ok=True)
        joblib.dump(model,          "models/xgb_fraud_model.pkl")
        joblib.dump(best_threshold, "models/best_threshold.pkl")
        joblib.dump(FEATURE_COLS,   "models/feature_cols.pkl")

        print("[INFO] Model saved → models/xgb_fraud_model.pkl")

    return model, best_threshold


if __name__ == "__main__":
    train()
