"""
model_training.py
-----------------
Trains a LightGBM (gradient boosting) churn classifier.
Compares 6 models via MLflow, tunes threshold, and saves the best.

Models evaluated:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. LightGBM  ← primary
  4. XGBoost
  5. GradientBoostingClassifier
  6. HistGradientBoosting (sklearn fast variant)
"""

import os
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    classification_report, fbeta_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from feature_engineering import FEATURE_COLS, TARGET_COL

warnings.filterwarnings("ignore")
SEED = 42


# ── Model zoo ─────────────────────────────────────────────────────────────────
def get_models():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced",
                                       random_state=SEED)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            max_depth=8, random_state=SEED, n_jobs=-1
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05,
            max_depth=6, num_leaves=63,
            scale_pos_weight=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=SEED, n_jobs=-1, verbose=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.05,
            max_depth=5, scale_pos_weight=5,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=SEED, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, random_state=SEED,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05,
            max_depth=5, random_state=SEED,
        ),
    }


def tune_threshold(model, X_val, y_val) -> float:
    proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_val, (proba >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def train(data_path: str = "data/customers_features.parquet"):
    df = pd.read_parquet(data_path)
    X  = df[FEATURE_COLS].fillna(0)
    y  = df[TARGET_COL]

    print(f"[DATA] Shape: {X.shape}  |  Churn rate: {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
    )

    models   = get_models()
    results  = {}
    best_auc = 0.0
    best_name, best_model, best_threshold = None, None, 0.5

    mlflow.set_experiment("churn_prediction")

    for name, model in models.items():
        print(f"\n[TRAIN] {name} …")
        with mlflow.start_run(run_name=name):
            model.fit(X_tr, y_tr)
            proba_val = model.predict_proba(X_val)[:, 1]
            auc_val   = roc_auc_score(y_val, proba_val)
            pr_val    = average_precision_score(y_val, proba_val)

            mlflow.log_metrics({"val_roc_auc": auc_val, "val_pr_auc": pr_val})
            print(f"  val ROC-AUC: {auc_val:.4f}  |  PR-AUC: {pr_val:.4f}")

            results[name] = {"auc": auc_val, "pr_auc": pr_val}

            if auc_val > best_auc:
                best_auc       = auc_val
                best_name      = name
                best_model     = model
                best_threshold = tune_threshold(model, X_val, y_val)

    # ── Evaluate best model on test set ───────────────────────────────────────
    print(f"\n[BEST] {best_name}  (val AUC = {best_auc:.4f})")
    proba_test = best_model.predict_proba(X_test)[:, 1]
    y_pred     = (proba_test >= best_threshold).astype(int)

    test_auc = roc_auc_score(y_test, proba_test)
    test_f1  = f1_score(y_test, y_pred)
    test_f2  = fbeta_score(y_test, y_pred, beta=2)

    print(f"\n{'='*60}")
    print(f"  Test ROC-AUC : {test_auc:.4f}")
    print(f"  Test F1      : {test_f1:.4f}")
    print(f"  Test F2      : {test_f2:.4f}")
    print(f"  Threshold    : {best_threshold:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred,
                                 target_names=["Retained", "Churned"]))

    # ── Model comparison table ─────────────────────────────────────────────────
    comparison = pd.DataFrame(results).T.sort_values("auc", ascending=False)
    comparison.to_csv("outputs/model_comparison.csv")
    print("\n[COMPARISON]\n", comparison.to_string())

    # ── Persist ───────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model,      "models/churn_model.pkl")
    joblib.dump(best_threshold,  "models/churn_threshold.pkl")
    joblib.dump(FEATURE_COLS,    "models/churn_feature_cols.pkl")

    scored = X_test.copy()
    scored["is_churn"]     = y_test.values
    scored["churn_proba"]  = proba_test
    scored["churn_pred"]   = y_pred
    scored.to_csv("outputs/test_scored.csv", index=False)

    print("[INFO] Model saved → models/churn_model.pkl")
    return best_model, best_threshold, best_name


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    train()
